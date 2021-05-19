**Smart Window**

This is a Planspiel project, a module offered by the Master's program Web Engineering at Technische Universität Chemnitz which focus on Modern Aspects of Software Development with development of "Startup Company". 

For this module we have created an web application named "Smart Window", which detects point of interest in real time and provides information of detected point of interest. Additionally, information like time, date,weather, station and their connecting trains are also integrated in the system.

**Installation Steps**

(Python version 3.9 doesnot install all the requirements therefore either upgarde all the requirements or use python 3.7)

1. Clone the repository 

    `git clone https://github.com/anijorshrestha/smartRail`
    

2. Go into the repository
   
   `cd <reponame>` 

3. Create a virtual environment inside the repo

    `pip install virtualenv` 

    `virtualenv venv`

4. Activate the environment

    `.\venv\Script\activate`

5. Install from requirements.txt

    `pip install -r requirements.txt`

6. All the necessary libraries will be installed. Then hit the following command to run in your browser
    
    `python manage.py runserver`

7. You will see something like this

    
    _Watching for file changes with StatReloader_

    _System check identified no issues (0 silenced)._

    _April 08, 2021 - 14:59:47_

    _Django version 3.1.4, using settings 'SmartRail.settings'_

    _Starting development server at http://127.0.0.1:8000/_

    _Quit the server with CTRL-BREAK._


8. Click or copy link (http://127.0.0.1:8000/) to see the working application

Thank You

![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)



