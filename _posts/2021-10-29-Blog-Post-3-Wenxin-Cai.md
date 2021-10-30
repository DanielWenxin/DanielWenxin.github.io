1. The app you’re going to build is a simple message bank. It should do two things:
   - Allow the user to submit messages to the bank.
   - Allow the user to view a sample of the messages currently stored in the bank.
2. Additionally, you should use CSS to make your app look attractive and interesting! I encourage you to get creative on this.
3. Your Blog Post should contain several screencaps from the functioning of your app, as well as a discussion of the Python methods you implemented to create your app.
4. You are free to (and indeed encouraged) to build on any of the example apps from class, as well as any other resources you are able to find.
5. The code for your app must be hosted in a GitHub repository. I suggest you begin by creating such a repository. Commit and push each time you successfully add a new piece of functionality or resolve a bug.


```python
# to run this website and watch for changes: 
# $ export FLASK_ENV=development; flask run

from flask import Flask, Blueprint, current_app, g, render_template, redirect, request, flash, url_for, session
from flask.cli import with_appcontext

from werkzeug.security import check_password_hash, generate_password_hash

import sqlite3
import click

import random
import string

app = Flask(__name__)
```

## Separate code blocks and explanations for each of the Python functions you used to build your app (there should be at least 5).

### 1. main( )


```python
@app.route('/')
def main():
    return render_template('main_better.html')   # To render my main_better.html template
```

Write the function main to render your main_better.html template that will be shown up on my main page of website

### 2. Submit( )


```python
@app.route('/Submit/', methods=['POST', 'GET'])
# since this page will both transmit and receive data
# you should ensure that it supports both POST and GET methods

def Submit():
    if request.method == 'GET':
        return render_template('Submit.html')
# In the GET case, we can just render the submit.html template with no other parameters
    if request.method == 'POST':
        try: 
            insert_message(request)
            # In the POST case, you should call insert_message()
            return render_template('Submit.html', thanks=True)
            # then render the submit.html template 
            # and the "thanks" will activate my greeting in the Submit.html
        except:
            return render_template('Submit.html', error=True)
            # then render the submit.html template 
            # and the "thanks" will activate my error message in the Submit.html
```

We are going to write a function to render_template() the submit.html template. Since this page will both transmit and receive data, we should ensure that it supports both POST and GET methods, and give it appropriate behavior in each one. 
In the GET case, we render the submit.html template with no other parameters. In the POST case, we call insert_message() (and then render the submit.html template).

### 3. View( )


```python
@app.route('/View/')
def View():
    message=random_messages(6)    
    # first call random_messages() to grab some random messages (I chose a cap of 6)
    return render_template('View.html',messages=message)
    # pass these messages as an argument to render_template()
    # and set the messages that we tend to print on the website as the message we grab from random_messages()
```

We are going to write a function to render the view.html template. This function should first call random_messages() to grab some random messages (I chose a cap of 5), and then pass these messages as an argument to render_template().

### 4. get_message_db( )


```python
def get_message_db():
    if 'message_db' not in g:
    # Check whether there is a database called message_db in the g attribute of the app
        g.message_db = sqlite3.connect('message_db.sqlite')
    # If not, then connect to that database

    cursor = g.message_db.cursor()
    cmd1 = """CREATE TABLE IF NOT EXISTS messages(id INTEGER,
                                                handle TEXT,
                                                message TEXT)"""
    cursor.execute(cmd1)
    # check whether a table called messages exists in message_db, and create it if not
    # Give the table an id column (integer), a handle column (text), and a message column (text)
    return g.message_db
    # Return the connection g.message_db
```

The function get_message_db() should handle creating the database of messages.
1. Check whether there is a database called message_db in the g attribute of the app. If not, then connect to that database, ensuring that the connection is an attribute of g. To do this last step, write a line like do g.message_db = sqlite3.connect("messages_db.sqlite)

2. Check whether a table called messages exists in message_db, and create it if not. For this purpose, the SQL command CREATE TABLE IF NOT EXISTS is helpful. Give the table an id column (integer), a handle column (text), and a message column (text).

3. Return the connection g.message_db.

### 5. insert_message(request)


```python
def insert_message(request):
    message = request.form['message']  # Extract the message and the handle from request
    handle = request.form['handle']    # Extract the message and the handle from request
    TB = get_message_db()
    cursor = TB.cursor()
    cmd2 = """SELECT COUNT(*) FROM messages"""
    cursor.execute(cmd2)
    # By applying cursor and the cmd2 to get the total number of rows in messages.
    TB.execute(f""" INSERT INTO messages(id, handle, message) VALUES ({cursor.fetchone()[0]+1}, "{handle}", "{message}");""")
    # Using a cursor, insert the message into the message database
    # To ensure that the ID number of each message is unique. 
    # One way to do this is by setting the ID number of a message equal to one plus the current number of rows in message_db.
    # Remember that we’ll need to provide an ID number, the handle, and the message itself. 
    # We’ll need to write a SQL command to perform the insertion.
    TB.commit()
    # It is necessary to run db.commit() after inserting a row into db in order to ensure that your row insertion has been saved.
    TB.close()
    # Don’t forget to close the database connection within the function!
```

The function insert_message(request) should handle inserting a user message into the database of messages.
1. Extract the message and the handle from request. We’ll need to ensure that your submit.html template creates these fields from user input by appropriately specifying the name of the input elements. 

2. Using a cursor, insert the message into the message database. Remember that we’ll need to provide an ID number, the handle, and the message itself. We’ll need to write a SQL command to perform the insertion.
  - Note: when working directly with SQL commands, it is necessary to run db.commit() after inserting a row into db in order to ensure that your row insertion has been saved.
  - We should ensure that the ID number of each message is unique. One way to do this is by setting the ID number of a message equal to one plus the current number of rows in message_db.
  - Don’t forget to close the database connection within the function!

### 6. random_messages(n)


```python
def random_messages(n):
    TB = get_message_db()     # Set TB as the g.message_db database
    cursor = TB.cursor()
    cmd4 = f"""SELECT handle,message FROM messages order by RANDOM() LIMIT {n}"""
    cursor.execute(cmd4)
    # We use a cursor adn SQL command to randomly select n handle and message from messages
    result = cursor.fetchall()
    TB.close()
    # Don’t forget to close the database connection within the function!
    return result
    # return a collection of n random messages
```

Write a function called random_messages(n) which will return a collection of n random messages from the message_db, or fewer if necessary.
Don’t forget to close the database connection within the function!

## A discussion of at least one of the template files you used in your app. You can copy the source code of the template file into your markdown post.


```python
{% extends 'base.html' %}        # We had the submit.html template extend base.html

{% block header %}
  <h1>{% block title %}Submit{% endblock %}</h1>     # Set up the title as "Submit"
{% endblock %}

{% block content %}             # Here is the html that creates the interactable elements
  <form method="post" enctype="multipart/form-data">
    <label for="message">Your message</label>     # Here is the block asking for the user to fill in their message
    <br>
    <input type="text" name="message" id="message"> 
    # It shows that the type of the input is "text", and we set their name and id as "message"
    <br>
    <label for="name">Your name or handle: </label> 
    # Here is the block asking for the user to fill in their users' name
    <br>
    <input type="text" name="handle" id="handle">
    # It shows that the type of the input is "text", and we set their name and id as "handle"
    <br>
    <input type="submit" value="Submit">
    # Here is the submit button created in out html
    <br>
  </form>

  {% if thanks %}
    # Recall the command--thanks=True--written in app.py
    # so if "thanks" shown up
    <br>
    Thank You For Submitting!
    # We'll print the greeting "Thank You For Submitting!" on our website
  {% endif %}

  {% if error %}
    # Recall the command--error=True--written in app.py
    # so if "error" shown up
    <br>
    Oh, we cannot read!
    # We'll print the error message "Oh, we cannot read!" on our website
  {% endif %}

{% endblock %}

```

## Your blog post must include two screencaps:

  - In the first screencap, you should show an example of a user submitting a message. In the handle field, please use either your name or the nickname used on your PIC16B blog. I’ve demonstrated this in my screencap illustrating the submission interface in Section §1.
{% include interface1.png %}

  - In the second screencap, you should show an example of a user viewing submitted messages. Show at least two messages, one of which is the message you submitted in the previous screencap. This message should show your name or nickname. I’ve demonstrated this in my screencap illustrating the viewing interface in Section §2.
{% include interface2.png %}

## Additionally, please include in your blog post a link to the GitHub repository containing the code for your app.

https://github.com/DanielWenxin/BLOG3/tree/main/flask-interactions-main2
