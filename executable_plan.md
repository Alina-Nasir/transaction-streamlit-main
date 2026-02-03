This is a complex architectural task because it involves cross-language processes (Python + C++) and strict environment management.

Here is the **Master Plan** you can hand directly to your Coding Agent. It breaks the process into 4 distinct phases: **Preparation**, **Code Logic**, **Bundling**, and **Installer Creation**.

---

### Phase 1: Environment & Asset Preparation

**Goal:** Centralize all dependencies so the installer doesn't rely on absolute paths on your development machine.

1. **Activate VENV:**
* *Instruction:* Ensure the terminal is running inside your root project's `venv`.
* *Action:* Run `pip install pyinstaller`. (This ensures PyInstaller uses the *exact* Python version and packages from your venv).


2. **Create "Build Assets" Folder:**
* *Instruction:* Create a folder named `build_assets` in the root directory.


3. **Localize Llama.cpp:**
* *Instruction:* Go to the path defined in your `start_llama_server.bat`. Copy the `llama-server.exe` (and any `.dll` files in that folder) into `build_assets/bin/`.


4. **Localize Model:**
* *Instruction:* Copy your `.gguf` model file into `build_assets/model/`.



---

### Phase 2: Python Code Logic

**Goal:** Replace the `.bat` file with a Python script that manages the database, the AI server, and the UI.

#### Step 2.1: Create `db_manager.py`

* **Role:** Handles the SQLite database without requiring a server installation.
* **Logic:**
* Connect to `sqlite3.connect('transactions_data.db')`.
* Create a table `extraction_logs` if it doesn't exist.
* Provide a function `insert_record(json_data)` to save the extracting results.



#### Step 2.2: Create `launcher.py` (The Entry Point)

* **Role:** This is the file PyInstaller will compile. It replaces your `.bat` file.
* **Critical Logic:**
1. **Path Resolver:** Define a function to detect if running in "Frozen" mode (compiled) or "Dev" mode using `sys._MEIPASS`.
2. **Server Start:** Use `subprocess.Popen` to launch the bundled `llama-server.exe` from the temp folder.
* *Note:* replicate the flags from your original `.bat` file here (e.g., context size, layers).


3. **Streamlit Start:** Instead of running `streamlit run app.py` (which fails in exe), use:
```python
from streamlit.web import cli as stcli
sys.argv = ["streamlit", "run", resource_path("app.py"), "--global.developmentMode=false"]
sys.exit(stcli.main())

```


4. **Cleanup:** Use a `try...finally` block to ensure `llama_process.kill()` is called when the user closes the app.



---

### Phase 3: PyInstaller Configuration (The Spec File)

**Goal:** Tell PyInstaller to bundle your specific `venv` packages and external binaries.

1. **Generate Spec:**
* *Command:* `pyi-makespec launcher.py`


2. **Edit `launcher.spec`:**
* **Datas:** Map your Python source files.
```python
datas=[
    ('app.py', '.'),
    ('db_manager.py', '.'),
    ('build_assets/model/your_model.gguf', 'model'), # Destination 'model' folder
]

```


* **Binaries:** Map the Llama.cpp server.
```python
binaries=[
    ('build_assets/bin/llama-server.exe', 'bin'), # Destination 'bin' folder
     # Include any .dlls from that folder here too if they exist
]

```


* **Hidden Imports:** Streamlit often hides dependencies. Add these to `hiddenimports`:
```python
hiddenimports=['streamlit', 'pandas', 'altair', 'sqlite3']

```




3. **Build:**
* *Command:* `pyinstaller launcher.spec --clean --noconfirm`
* *Result:* A folder `dist/launcher` containing the executable and all `venv` libraries.



---

### Phase 4: Installer Creation (Inno Setup)

**Goal:** Compress the `dist` folder into a single setup file for the client.

* *Prerequisite:* Install "Inno Setup Compiler" (standard Windows tool).
* **Script Logic (`setup.iss`):**
1. **Source:** Point `Source: "dist\launcher\*"` to `DestDir: "{app}"`.
2. **Flags:** Use `recursesubdirs` to maintain the folder structure inside Program Files.
3. **Icons:** Create a shortcut pointing to `"{app}\launcher.exe"`.
4. **Permissions:** (Optional) If writing to DB in the install folder, add `Permissions: users-modify` to the `DestDir`. *Better practice: Write DB to user's AppData folder in python code.*



### Summary Checklist for the Coding Agent

* [ ] **Strictness:** Do not upgrade or change packages. Use `pip freeze` to verify environment before building.
* [ ] **Paths:** Use `os.path.join(sys._MEIPASS, "relative/path")` for *everything* inside `launcher.py`.
* [ ] **Llama:** Ensure `llama-server.exe` is executable and not blocked by permissions in the build folder.
* [ ] **DB:** Ensure the SQLite DB is saved to a writable location (like `%APPDATA%`) and not the `Program Files` directory, or else the app will crash due to "Permission Denied" on client machines.