from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import login_user, logout_user
from werkzeug.security import check_password_hash
from pymongo import MongoClient



auth = Blueprint('auth', __name__)
client = MongoClient('localhost', 27017)
db = client.flask_db
users = db.users

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        unhashed_password = request.form['password']
        userType = request.form['userType']
        if userType == '0':
            boolUserType = 0
        else:
            boolUserType = 1
        user = {
            "_id" : users.count_documents({})+1,
            "name": name, 
            "password": unhashed_password,
            "admin": False,  
            "expert": boolUserType
        }

        users.insert_one(user)
        return redirect(url_for('routes.login'))
    
    return render_template('register.html')

# @auth.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         name = request.form['name']
#         password = request.form['password']

#         # user = users.find({"name" : name})
#         print("---------------------------------------------------------")
#         print(users.count_documents({"name" : name}))
#         error_message = ''

#         if users.count_documents({"name" : name}) > 0:
#             user = users.find({"name" : name})
#         else:
#             error_message = 'User not Found.'

#         if not error_message and user[0]['password'] != password:
#             error_message = 'Could not login. Please check and try again.'

#         if not error_message:
#             login_user(user)
#             return redirect(url_for('main.index'))

#     return render_template('login.html')

# @auth.route('/logout')
# def logout():
#     logout_user()
#     return redirect(url_for('auth.login'))