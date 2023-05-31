from flask import Flask, Blueprint, redirect, session
from flask_login import LoginManager
from flask import render_template, url_for, request, flash
from .form import Login
from flask import request
from werkzeug.urls import url_parse
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import current_user, login_user, logout_user, login_required
from pymongo import MongoClient

routes = Blueprint('routes', __name__)
login = LoginManager(routes)
login.login_view = 'login'

client = MongoClient('localhost', 27017)
db = client.flask_db
users = db.users
Quizes = db.quizes

class User:
    def __init__(self, username, expert):
        self.username = username
        self.is_authenticated = True
        self.expert = expert

    # @staticmethod
    # def is_authenticated():
    #     return True

    @staticmethod
    def get_userId():
        return 1

    @staticmethod
    def is_active():
        return True

    @staticmethod
    def is_anonymous():
        return False

    def get_id(self):
        return self.username

    @staticmethod
    def check_password(password_hash, password):
        return password_hash == password


    @login.user_loader
    def load_user(username):
        u = users.find_one({"name": username})
        if not u:
            return None
        return User(username=u['name'])


    @routes.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            # if current_user.is_authenticated:
                # return redirect(url_for('main.home'))
            # form = Login()
            # if form.validate_on_submit():
            name = request.form['name']
            password = request.form['password']
            user = users.find_one({"name": name})
            if user and User.check_password(user['password'], password):
                user_obj = User(username=user['name'], expert=user['expert'])
                # login_user(user_obj)
                session['user_id'] = user['_id']
                session['username'] = name
                session['is_expert'] = user['expert']
                session['progressBar'] = '0'
                if user['expert']:
                    session['quizPart'] = 1
                    session['QuizCreated'] = False
                    session['QuizId'] = 0
                session['is_admin'] = user['admin']
                session['is_authenticated'] = True

                next_page = request.args.get('next')
                print("YEEEEEPPPPPPP___________________________")
                print(user_obj.is_authenticated)
                if not next_page or url_parse(next_page).netloc != '':
                    next_page = url_for('main.home')
                return redirect(next_page)
            else:
                flash("Invalid username or password")
                return render_template('login.html')
        return render_template('login.html')

    @routes.route('/logout')
    def logout():
        if session['is_expert'] and session['QuizCreated'] == False:
            Quizes.delete_one( { "QuizId": session['QuizId'] } )
        session.clear()
        return redirect(url_for('main.index'))