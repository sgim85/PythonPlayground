Source: [Python in VS Code](https://code.visualstudio.com/docs/python/python-quick-start)

### Steps to create python project
* Create project folder
* Navigate to the folder via command line and enter "*Code .*" to open VS Code.
* Enable virtual environment. Open command pallette (*Ctrl + Shift + P*). Type "Python: Create Environment". Select "Venv", then select latest python version installed.
    * A virtual environment isolates projects so any installed libraries don't interfere with other project envs or the global interpreter env.
    * An environment folder (/.venv) will be created in the project root.
    
To Run the file in focus, open the Command Pallete and enter "*Run Python File*"

### To Debug, use following commands
* F5 - Run Debugger
* F10 - Step over
* F11 - Step into
* Shift+F11 - Step out
* Shit+F5 - Terminate debugger
* Ctrl+Shift+F5 - Restart debugger

### Installing Python Packages
*pip install* is the standard command used to install Python packages from the Python Package Index (PyPI) and other indexes.
Use *pip install <package_name>* to install a library, or *pip install -r requirements.txt* to install dependencies listed in a file.

While not strictly mandatory in all cases, it is highly recommended to use *python -m pip* instead of just *pip*. Prepending *python -m* ensures you are using the pip associated with the specific Python interpreter you are running, preventing issues with multiple Python versions.

To install packages in *requirements.txt* and additional packages, use command *pip install -r requirements.txt <package_1> <package_2>..*

### Managing dependencies across environments
When working on Python projects, it’s essential to manage your dependencies effectively. One useful tip is to use the *pip freeze > requirements.txt* command.
This command helps you create a *requirements.txt* file that lists all the packages installed in your virtual environment. This file can then be used to recreate the same environment elsewhere.
Furthermore, you can continue to add dependencies to it as your project may grow in complexity using command "*pip install -r requirements.txt*."
By following these steps, you ensure that your project dependencies are consistent across different environments, making it easier to collaborate with others and deploy your project.

### Useful shortcuts
* Ctrl+R: Helps switch between projects by opening a list of recently opened folders and workspaces.
* Ctrl+Shift+`: Open command prompt for your selected terminal
* Ctrl+Shift+P: Open VS code command pallette
* ctrl+Shift+x: Open extensions
* Shift+Alt+F: Format code
* Editing shortcuts:
    * F12: Go to Definition
    * Alt+F12: Peek Definition

### Read-Eval-Print Loop
* To activate the classic Python REPL, type "python" in the terminal, which will switch to ">>>". OR you can use command palette then enter "**Python: Start Terminal REPL**"
* To activate the native VS Code REPL, which has additional VS Code enhancements like intellicense and syntax highlighting, you have a couple of options:
    * Search for **Python: Start Native REPL** in the command palette.
    * Via Smart Send (*Shift+Enter*) and "Run Selection/Line in Python REPL" by setting "python.REPL.sendToNativeREPL": true in your settings.json file. NOTE: You can opt to continue to use the REPL built-in to Python located in the terminal ( >>> ) by setting "python.REPL.sendToNativeREPL": false in your settings.json.

### Jupyter Notebooks
Jupyter is an open-source project that lets you easily combine Markdown text and executable Python source code on one canvas called a **notebook**. [Learn more](https://code.visualstudio.com/docs/python/jupyter-support-py) about Jupyter support in VS Code.

### Miscellaneous
* VS code comes with a "Python Profile" that you can enable. A good starting point for Python development as it sets up a lot of prerequisites: extensions, libraries, settings, etc.
* **Smart Send** feature (*Shift+Enter*): Smart Send looks at the code where the cursor is placed, sends the smallest runnable chunk of code to the Python REPL, and then places your cursor at the next line of code. This enables you to easily and efficiently run Python code in your program.
* To run python file, press play button in the top right (*Run Python File*). Alternatively, right-click inside file, then go to **Run Python > Run > Python File in Terminal**
* To run selected lines, press *Shift+Enter* (Smart Send). Alternatively, right-click inside file, then go to **Run Python > Run Selection/Line in Python Terminal**
* Find editing tips provided by *Pylance* — the default language server for Python in VS Code that's installed alongside the Python extension to provide IntelliSense features — [here](https://code.visualstudio.com/docs/python/editing).
* The *settings.json* stores user preferences and customizations for the editor. E.g. to specify intellisense behavior for a language. Two ways to access settings:
    * Command palette: Type *Open Settings JSON*
    * Editor UI: *File > Preferences > Settings*
* **import module** vs **from module import**: 
    * **import module:** imports the entire module; e.g. *import math*, which is then used as *math.sqrt*, *math.pi*, and so on. 
    * **from module import:** imports specific items (functions, classes, or variables) from a module; e.g. *from math import sqrt, pi*, with those imported functions used direcly...*sqrt(), pi* and so on.
* **package** vs **distribution**: a *package* refers to a directory structure for organizing code, while a *distribution* (or distribution package) is the installable software bundle used for sharing that code.
* **Workspace settings.json vs User settings.json**: Workspace settings override user settings for specific projects. They ensure consistent configurations (linters, formatters) across teams by sharing project-specific settings. They are highly prioritized and can be edited directly or via the settings. Workspace settings can be updated via command palette (*Preferences: Open Workspace Settings (JSON)*); for user settings use command *Preferences: Open User Settings (JSON)*
* **Linting**: Syntax error detection is enabled by default in the Python extension's Language Server. You can enable linting for additional code detection, including stylistic checks, by installing a *Linting* extension, such as *flake8*.
* **Formatting**: Formatting makes source code easier to read by human beings. By enforcing particular rules and conventions such as line spacing, indents, and spacing around operators, the code becomes more visually organized and comprehensible. Formatting extensions you can install include *autopep8* and *Black formatter*. 
    * To set a default formatter, right click a *.py* file, select *Format Document With...*, select *Configure Default Formatter*; alternatively, you can set the default formatter by setting "editor.defaultFormatter" in the *settings.json* file, which can be opened in command palette via *Preferences: Open User Settings (JSON)*.
    * To format code, right-click on the editor and select *Format Document*, or by using the *Shift+Alt+F* shortcut.
    * To enable auto-formatting on file save, enable *editor.formatOnSave* in user settings.
* [Python in a container](https://code.visualstudio.com/docs/containers/quickstart-python)
* **Dev Frameworks**
    * [Django](https://code.visualstudio.com/docs/python/tutorial-django)
    * [FastAPI](https://code.visualstudio.com/docs/python/tutorial-fastapi)
    * [Flask](https://code.visualstudio.com/docs/python/tutorial-flask)
* [Run and Debug Python in the Web](https://code.visualstudio.com/docs/python/python-web) (experimental support)
* **Tests**: Python has a built-in *unittest* framework. A thirdparty alternative is the *pytest* framework which offers more features, if desired.
    * Tests can be configured anytime by using the **Python: Configure Tests** command from the Command Palette or by setting either **python.testing.unittestEnabled** or **python.testing.pytestEnabled** in the Settings editor.
    * *Test Discovery*: By default, the Python extension attempts to discover tests once you enable a framework. You can trigger test discovery at any time using the Test: Refresh Tests command from the Command Palette.
    * [More on testing](https://code.visualstudio.com/docs/python/testing)
* **with** statement is a control flow structure that simplifies resource management by automatically handling the setup and teardown phases of common operations. It is used with objects that support the context manager protocol, which guarantees that a resource is properly released, even if errors occur within the associated code block. It is similar to C#'s *using* statement.

            
                # Without 'with' (requires explicit closing)
                file = open("example.txt", "w")
                try:
                    file.write("Hello, World!")
                finally:
                    file.close()

                # With 'with' (automatic closing)
                with open("example.txt", "w") as file:
                    file.write("Hello, World!")
            
* *def main* is a convention, not a requirement, used to define a function that acts as the entry point for a script when it is run directly. This is used in conjuction with a special built-variable **\_\_name__**.

            
                def main():
                    # Main function to run the program logic.
                    print("This code runs when the script is executed directly.")
                    
                    # Other code here

                # Call the main() function only when the script is executed as the primary program
                if __name__ == "__main__":
                    main()

* **Environments:** A workspace can have multiple projects, and environments can be configured at project level instead of workspace level. A Python project is any file or folder you want to associate with a specific environment. By default, your entire workspace uses one environment. Projects let you assign different environments to different folders. This is essential for mono-repos, microservices, or testing across Python versions.

            Workspace
            ├── Python Project: backend/
            │     └── Environment: .venv (Python 3.12)
            │            └── Manager: venv
            │
            ├── Python Project: frontend-utils/
            │     └── Environment: .venv (Python 3.10)
            │            └── Manager: venv
            │
            └── Python Project: ml-pipeline/
                └── Environment: ml-env (Python 3.11)
                        └── Manager: conda

    * [Learn more on environments](https://code.visualstudio.com/docs/python/environments)
* **Debugging:** [Learn more on debugging, including advanced debugging](https://code.visualstudio.com/docs/python/debugging)


**Typical Python Project Structure**

        my_project/  
        ├── src/  
        │   ├── my_package/  
        │   │   ├── __init__.py  
        │   │   ├── module1.py  
        │   │   └── utils.py  
        ├── tests/  
        │   ├── test_module1.py  
        │   └── test_utils.py  
        ├── .gitignore  
        ├── pyproject.toml  
        ├── requirements.txt  
        ├── README.md  
        └── LICENSE  
