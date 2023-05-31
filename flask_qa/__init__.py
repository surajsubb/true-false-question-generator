from flask import Flask, render_template, request, url_for, redirect
from pymongo import MongoClient
from flask_login import LoginManager

from .routes.auth import auth
from .routes.main import main
from .routes.routes import routes, User

def create_app():
    app = Flask(__name__)
    app.config.update(dict(
        SECRET_KEY="powerful secretkey",
        WTF_CSRF_SECRET_KEY="a csrf secret key"
    ))
    client = MongoClient('localhost', 27017)
    db = client.flask_db
    users = db.users

    login_manager = LoginManager()
    login_manager.init_app(app)

    login_manager.login_view = 'auth.login'

    @login_manager.user_loader
    def load_user(user_id):
        return User.get_userId()

    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(routes)

    return app