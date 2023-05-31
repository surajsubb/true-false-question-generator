from flask import Flask, Blueprint, render_template, request, redirect, url_for, session
from flask_login import current_user, login_required
from bson.json_util import dumps
from pymongo import MongoClient
from removeNewline import remove_newline
from samplingAlgo import sampleQuestions
from concurrent.futures import ThreadPoolExecutor

import json
import os
import re
import random
import time
import asyncio
import hashlib

_executor = ThreadPoolExecutor(1)

app = Flask(__name__)

main = Blueprint('main', __name__)
client = MongoClient('localhost', 27017)
db = client.flask_db
users = db.users
Question = db.questions
Scores = db.studentScores
Quizes = db.quizes

UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = set(['pdf'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@main.route('/')
def index():
    return render_template('landingPage.html')


@main.route('/home')
def home():
    quizes_cursor = Quizes.find()
    scores_cursor = Scores.find({"user_id": session['user_id']})

    quizes_string = dumps(list(quizes_cursor))
    all_quizes = json.loads(quizes_string)

    scores_string = dumps(list(scores_cursor))
    scores = json.loads(scores_string)

    quizes = []
    for quiz in all_quizes:
        score = Scores.find_one({"quizId": quiz["QuizId"]})
        if score:
            continue
        else:
            quizes.append(quiz)    
    context = {
        'quizes' : quizes
    }

    return render_template('home.html', **context)

@main.route('/quiz/<quiz_id>' , methods=['GET', 'POST'])
def quiz(quiz_id):
    if not session['is_authenticated'] and not session['is_expert']:
        return redirect(url_for('main.index'))
  
    questions = Quizes.find_one({"QuizId": int(quiz_id)})
    if request.method == 'POST':
        number_of_questions = len(questions['Questions'])
        answers = []
        
        for i in range (0, number_of_questions):
            answers.append(request.form[str(i+1)+'answer'])
        
        i = 0
        count = 0
        for answer in answers:
            if re.sub(r'\d', '', answer) == questions["Questions"][i]['Answer'].lower():
                count+=1
            i+=1
        finalScore = count
        score = {
            'user_id' : session['user_id'],
            'username' : session['username'],
            'quizId' : questions['QuizId'],
            'subject' : questions['Subject'],
            'topic' : questions['QuizTopic'],
            'score' : finalScore
        }
        Scores.insert_one(score)
        return redirect(url_for('main.scores'))



    count = 1
    Questions = questions["Questions"]
    
    jsonString = json.dumps(Questions)
    checksum = hashlib.md5(jsonString.encode("utf-8")).hexdigest()
    
    if questions['checksum'] != checksum:
        session.clear()
        return redirect(url_for('routes.login'))
        
    for question in Questions:
        questions["Questions"][count-1]["id"] = count
        count+=1
    context = {
        'questions' : questions
    }

    return render_template('quiz.html', **context)

@main.route('/scores')
def scores():
    if not session['is_authenticated'] and not session['is_expert']:
        return redirect(url_for('main.index'))
    
    allScores = Scores.find({"user_id": session['user_id']})
    print(allScores)
    context ={
        "Scores" : allScores
    }
    return render_template('scores.html', **context)

@main.route('/answerKey/<quiz_id>')
def answerKey(quiz_id):
    questions = Quizes.find_one({"QuizId": int(quiz_id)})
    context = {
        "Questions" : questions['Questions'],
        "Subject" : questions['Subject'],
        "Topic" : questions['QuizTopic']

    }
    return render_template('answerKey.html', **context)



@main.route('/createQuiz', methods=['GET', 'POST'])
def createQuiz():
    if not session['is_authenticated'] and session['is_expert']:
        return redirect(url_for('main.index'))   
    
    if request.method == 'POST':
        if session['quizPart'] == 1:
            session['QuizCreated'] = False
            session['QuizId'] = random.randint(0, 1000000000)
            session['Grade'] = request.form['grade']
            session['Subject'] = request.form['subject']
            session['Topic'] = request.form['topic']
            quiz = {
                "Grade": request.form['grade'],
                "Subject": request.form['subject'], 
                "Teacher": session['username'],
                "QuizTopic": request.form['topic'], 
                "QuizId": session['QuizId']
                
            }
            Quizes.insert_one(quiz)
            session['quizPart'] = 2
            return redirect(url_for('main.createQuiz'))

        elif session['quizPart'] == 2:
            session['progressBar'] = 0
            session['noOfQuestions'] = request.form['noOfQuestions']
            session['easyQuestions'] = request.form['easyQuestions']
            session['mediumQuestions'] = request.form['mediumQuestions']
            session['hardQuestions'] = request.form['hardQuestions']
            
            file = request.files['pdf-file']
            filename = session['username'] + '_' + str(session['QuizId']) +'.pdf'
            filepath = os.path.join(app.config['UPLOAD_FOLDER']+'/pdfs', filename)
            if file and allowed_file(file.filename):
                file.save(filepath)
            cmd = 'java -jar getText.jar ./pdfs/' + session['username'] + '_' + str(session['QuizId']) + '.pdf >> ./texts/directFromPDF.txt'
            os.system(cmd)
            remove_newline()
            cmd = 'node textCorpus-preprocessing.js'
            os.system(cmd)
            cmd = 'python3 neural_coref/neural_coref.py'
            os.system(cmd)
            cmd = 'mv texts/tempneuralcoreferenced.txt texts/neuralcoreferenced_'+session['Grade']+'_'+session['Subject']+'_'+session['Topic']+'.txt'
            os.system(cmd)
            session['quizPart'] = 3
            return redirect(url_for('main.createQuiz'))
        
        elif session['quizPart'] == 3:
            cmd = 'python3 FYP-20230413T131216Z-001/FYP/Summarization/summarize.py --file_location "texts/neuralcoreferenced_' + session['Grade'] + '_' + session['Subject'] + '_' + session['Topic'] + '.txt" --glove_location "FYP-20230413T131216Z-001/FYP/Summarization/glove.6B.100d.txt" --save_location "texts"'
            os.system(cmd)
            cmd = 'python3 FYP-20230413T131216Z-001/FYP/Answer\ Identification/run_main.py --mode "test" --Unlabeled_dataset "texts/Summary/neuralcoreferenced_' + session['Grade'] + '_' + session['Subject'] + '_' + session['Topic'] + '-Summarized.txt" --model_name "bert-base-cased" --model_type "bert" --save_location "FYP-20230413T131216Z-001/FYP/Answer Identification/saved_model"'
            os.system(cmd)
            time.sleep(3)
            sampleQuestions('jsons/Generated_Questions.json', 'Class' + str(session['Grade']), session['Subject'], session['Topic'], int(session['noOfQuestions']), float(session['easyQuestions']), float(session['mediumQuestions']), float(session['hardQuestions']) )
            print('Sampling of Questions Has been Done!')

            topic = session['Topic']
            with open('curatedQuestionSet/sampleQuestions-'+ session['Grade'] + '-' + session['Subject'] + '-' + topic.lower() +'.json') as json_file:
                data = json.load(json_file)
            questions = []
            count = 1
            for question in data:
                question['id'] = count
                questions.append(question)
                count+=1

            print(questions)
            context = {
                'questions' : data
            }
            return render_template('editQuiz.html', **context)
            
    return render_template('createQuiz.html')

@main.route('/editQuiz' , methods=['GET', 'POST'])
def editQuiz():
    if not session['is_authenticated'] and session['is_expert']:
        return redirect(url_for('main.index'))  

    noOfQuestions  = int(session['noOfQuestions'])
    Questions = []
    Question = {}
    for i in range (1,noOfQuestions+1):
        Question['Statement'] = request.form[str(i)+'question']
        Question['Answer'] = request.form[str(i)+'answer']
        Question['id'] = i
        Questions.append(Question)
        Question = {}

    print(Questions)
    jsonString = json.dumps(Questions)
    checksum = hashlib.md5(jsonString.encode("utf-8")).hexdigest()

    Quizes.update_one( {"QuizId": session["QuizId"]}, {"$set" : { "Questions" : Questions, "checksum" :  checksum}} )
    session['QuizCreated'] = True
    return render_template('home.html')

@main.route('/getProgressPreProcessing')
def getProgressPreProcessing():
    if(os.path.isfile(UPLOAD_FOLDER+'/texts/neuralcoreferenced.txt')):
        return '110'
    elif(os.path.isfile(UPLOAD_FOLDER+'/texts/regexed.txt')):
        return '75'
    elif(os.path.isfile(UPLOAD_FOLDER+'/texts/directFromPDFWithoutNewline.txt')):
        return '50'
    elif(os.path.isfile(UPLOAD_FOLDER+'/texts/directFromPDF.txt')):
        return '25'
    else:
        return '0'

@main.route('/getProgressMLModels')
def getProgressMLModels():
    # if(os.path.isfile(UPLOAD_FOLDER+'/texts/neuralcoreferenced.txt')):
    #     return '110'
    # elif(os.path.isfile(UPLOAD_FOLDER+'/texts/regexed.txt')):
    #     return '75'
    if(os.path.isfile(UPLOAD_FOLDER+'/jsons/Answerpredictions.json')):
        return '75'
    elif(os.path.isfile(UPLOAD_FOLDER+'/texts/Summary/neuralcoreferenced_'+session['Grade'] + '_' + session['Subject'] + '_' + session['Topic'] + '-Summarized.txt"')):
        return '50'
    else:
        return '0'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS