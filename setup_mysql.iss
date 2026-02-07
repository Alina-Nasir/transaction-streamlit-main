; Inno Setup Script for Pakistani Bank Transaction Parser
; This creates a Windows installer that bundles everything needed including MySQL configuration

#define MyAppName "Pakistani Bank Transaction Parser"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Pakistan Bank Parser Team"
#define MyAppExeName "PakistanBankParser.exe"
#define MyAppURL "https://github.com/yourusername/transaction-parser"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{A8F4B2C1-9E3D-4F7A-8B2C-1D4E5F6A7B8C}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
; Output directory for the installer
OutputDir=installer_output
OutputBaseFilename=PakistanBankParser_Setup_v{#MyAppVersion}
; Compression settings
Compression=lzma2/ultra64
SolidCompression=yes
; Windows Vista or higher required
MinVersion=6.1
; Architecture
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Privileges
PrivilegesRequired=lowest
; Icon (optional - add if you have one)
; SetupIconFile=icon.ico
; UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main executable
Source: "dist\PakistanBankParser\PakistanBankParser.exe"; DestDir: "{app}"; Flags: ignoreversion

; All files from _internal directory (includes Python runtime, DLLs, models, everything)
Source: "dist\PakistanBankParser\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; Include documentation and batch processor configs
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion; DestName: "README.txt"
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "QUICKSTART.md"; DestDir: "{app}"; Flags: ignoreversion; DestName: "QUICKSTART.txt"
Source: "batch_config.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "batch_processor_service.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "batch_processor_start.py"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

; Desktop shortcut (if user selected it)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Option to run the application after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up configuration from AppData on uninstall (optional)
Type: filesandordirs; Name: "{userappdata}\PakistanBankParser"

[Code]
{ Custom installation with MySQL configuration and folder selection }

var
  BankSlipsFolder: string;
  FolderSelectionPage: TInputDirWizardPage;
  MySQLConfigPage: TInputQueryWizardPage;
  MySQLHost, MySQLPort, MySQLDatabase, MySQLUser, MySQLPassword: string;

procedure InitializeWizard();
begin
  { Create custom page for MySQL configuration }
  MySQLConfigPage := CreateInputQueryPage(wpSelectDir,
    'MySQL Database Configuration',
    'Configure MySQL connection for storing transaction data',
    'Please provide your MySQL server credentials. The installer will validate the connection before proceeding.' + #13#10 + #13#10 +
    'Note: Make sure MySQL server is running on your system.');
  
  { Add input fields for MySQL credentials }
  MySQLConfigPage.Add('MySQL Host:', False);
  MySQLConfigPage.Add('MySQL Port:', False);
  MySQLConfigPage.Add('Database Name:', False);
  MySQLConfigPage.Add('MySQL Username:', False);
  MySQLConfigPage.Add('MySQL Password:', True);  // Password field (masked)
  
  { Set default values }
  MySQLConfigPage.Values[0] := 'localhost';
  MySQLConfigPage.Values[1] := '3306';
  MySQLConfigPage.Values[2] := 'transactions_db';
  MySQLConfigPage.Values[3] := 'root';
  MySQLConfigPage.Values[4] := '';
  
  { Create custom page for folder selection }
  FolderSelectionPage := CreateInputDirPage(MySQLConfigPage.ID,
    'Bank Slips Folder',
    'Where should the batch processor monitor for new bank slips?',
    'Select a folder where you will drop bank slip images for automatic processing.' + #13#10 + #13#10 +
    'Note: The folder must already exist before you select it.',
    False,
    '');
  
  { Add a default suggestion path }
  FolderSelectionPage.Add('');
  FolderSelectionPage.Values[0] := ExpandConstant('{userdocs}\BankSlips\incoming');
end;

function ValidateMySQLConnection(Host, Port, Database, User, Password: string): Boolean;
var
  ResultCode: Integer;
  PythonExe: string;
  ValidateScript: string;
  ErrorMsg: TArrayOfString;
  OutputMsg: AnsiString;
  TempFile: string;
begin
  Result := False;
  
  { Use the bundled Python executable }
  PythonExe := ExpandConstant('{app}\_internal\python.exe');
  ValidateScript := ExpandConstant('{app}\_internal\validate_mysql.py');
  
  { Check if Python exe exists }
  if not FileExists(PythonExe) then
  begin
    MsgBox('Python executable not found. Please ensure installation files are complete.', mbError, MB_OK);
    Exit;
  end;
  
  { Check if validation script exists }
  if not FileExists(ValidateScript) then
  begin
    MsgBox('MySQL validation script not found. Please ensure installation files are complete.', mbError, MB_OK);
    Exit;
  end;
  
  { Create temporary file for output }
  TempFile := ExpandConstant('{tmp}\mysql_validation.txt');
  
  { Execute validation script }
  if Exec(PythonExe, 
          '"' + ValidateScript + '" "' + Host + '" "' + Port + '" "' + Database + '" "' + User + '" "' + Password + '" > "' + TempFile + '" 2>&1',
          '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    { Read output }
    if LoadStringFromFile(TempFile, OutputMsg) then
    begin
      if ResultCode = 0 then
      begin
        MsgBox('MySQL connection validated successfully!' + #13#10 + 
               'Database "' + Database + '" is ready.', mbInformation, MB_OK);
        Result := True;
      end
      else
      begin
        MsgBox('MySQL connection failed!' + #13#10 + #13#10 +
               'Details: ' + OutputMsg + #13#10 + #13#10 +
               'Please check your credentials and ensure MySQL server is running.', mbError, MB_OK);
        Result := False;
      end;
    end;
    
    { Clean up temp file }
    DeleteFile(TempFile);
  end
  else
  begin
    MsgBox('Failed to execute MySQL validation script.', mbError, MB_OK);
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  
  { Validate MySQL configuration }
  if CurPageID = MySQLConfigPage.ID then
  begin
    { Get MySQL credentials }
    MySQLHost := MySQLConfigPage.Values[0];
    MySQLPort := MySQLConfigPage.Values[1];
    MySQLDatabase := MySQLConfigPage.Values[2];
    MySQLUser := MySQLConfigPage.Values[3];
    MySQLPassword := MySQLConfigPage.Values[4];
    
    { Validate inputs }
    if (MySQLHost = '') or (MySQLPort = '') or (MySQLDatabase = '') or (MySQLUser = '') then
    begin
      MsgBox('Please fill in all MySQL connection details.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
    
    { Validate MySQL connection }
    Result := ValidateMySQLConnection(MySQLHost, MySQLPort, MySQLDatabase, MySQLUser, MySQLPassword);
  end;
  
  { Validate folder selection }
  if CurPageID = FolderSelectionPage.ID then
  begin
    { Get the selected folder }
    BankSlipsFolder := FolderSelectionPage.Values[0];
    
    { Validate folder path }
    if BankSlipsFolder = '' then
    begin
      MsgBox('Please select a valid folder path.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
    
    { Check if folder exists }
    if not DirExists(BankSlipsFolder) then
    begin
      MsgBox('The selected folder does not exist.' + #13#10 +
             'Please create the folder first and then select it.' + #13#10 +
             'Selected path: ' + BankSlipsFolder, mbError, MB_OK);
      Result := False;
      Exit;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ConfigPath: string;
  ConfigContent: string;
  AppDataPath: string;
  EscapedPath: string;
  EscapedPassword: string;
  i: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    { Create config directory }
    AppDataPath := ExpandConstant('{userappdata}\PakistanBankParser\config');
    ForceDirectories(AppDataPath);
    
    { Escape backslashes for JSON in folder path }
    EscapedPath := '';
    for i := 1 to Length(BankSlipsFolder) do
    begin
      if BankSlipsFolder[i] = '\' then
        EscapedPath := EscapedPath + '\\'
      else
        EscapedPath := EscapedPath + BankSlipsFolder[i];
    end;
    
    { Escape special characters in password for JSON }
    EscapedPassword := '';
    for i := 1 to Length(MySQLPassword) do
    begin
      if MySQLPassword[i] = '\' then
        EscapedPassword := EscapedPassword + '\\'
      else if MySQLPassword[i] = '"' then
        EscapedPassword := EscapedPassword + '\"'
      else
        EscapedPassword := EscapedPassword + MySQLPassword[i];
    end;
    
    { Create MySQL database configuration file }
    ConfigPath := AppDataPath + '\db_config.json';
    ConfigContent := 
      '{' + #13#10 +
      '  "type": "mysql",' + #13#10 +
      '  "host": "' + MySQLHost + '",' + #13#10 +
      '  "port": ' + MySQLPort + ',' + #13#10 +
      '  "database": "' + MySQLDatabase + '",' + #13#10 +
      '  "user": "' + MySQLUser + '",' + #13#10 +
      '  "password": "' + EscapedPassword + '"' + #13#10 +
      '}';
    
    { Write MySQL config file }
    SaveStringToFile(ConfigPath, ConfigContent, False);
    
    { Create batch processing configuration file }
    ConfigPath := AppDataPath + '\batch_config.json';
    ConfigContent := 
      '{' + #13#10 +
      '  "incoming_folder": "' + EscapedPath + '",' + #13#10 +
      '  "debounce_delay": 3.0,' + #13#10 +
      '  "enabled": true,' + #13#10 +
      '  "inference_timeout": 300' + #13#10 +
      '}';
    
    { Write batch config file }
    SaveStringToFile(ConfigPath, ConfigContent, False);
    
    MsgBox('Installation complete!' + #13#10 + #13#10 + 
           'MySQL Database Configuration:' + #13#10 +
           '• Host: ' + MySQLHost + ':' + MySQLPort + #13#10 +
           '• Database: ' + MySQLDatabase + #13#10 +
           '• User: ' + MySQLUser + #13#10 + #13#10 +
           'Batch Processing Configuration:' + #13#10 +
           '• Incoming Folder: ' + BankSlipsFolder + #13#10 + #13#10 +
           'How it works:' + #13#10 +
           '• Drop bank slip images in the configured folder' + #13#10 +
           '• The system will automatically process them' + #13#10 +
           '• Processed files remain in the same folder' + #13#10 +
           '• Results are saved to MySQL database' + #13#10 + #13#10 +
           'Important Notes:' + #13#10 +
           '• First launch may take 15-20 seconds to initialize' + #13#10 +
           '• The AI model will start automatically' + #13#10 +
           '• Your browser will open with the application' + #13#10 +
           '• Configuration stored in: %APPDATA%\PakistanBankParser' + #13#10 + #13#10 +
           'For help, see QUICKSTART.txt in the installation folder.',
           mbInformation, MB_OK);
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  MsgBox('Welcome to Pakistani Bank Transaction Parser Setup!' + #13#10 + #13#10 +
         'This installer includes:' + #13#10 +
         '• Complete Python runtime (no Python installation needed)' + #13#10 +
         '• AI vision model (Qwen3-VL-2B, ~1.5GB)' + #13#10 +
         '• llama.cpp inference engine' + #13#10 +
         '• MySQL database connector' + #13#10 +
         '• All required libraries' + #13#10 +
         '• Automatic batch processing service' + #13#10 + #13#10 +
         'Prerequisites:' + #13#10 +
         '• MySQL Server must be installed and running on your system' + #13#10 +
         '• You will need MySQL credentials to configure the database' + #13#10 + #13#10 +
         'No internet connection required after installation!' + #13#10 + #13#10 +
         'Installation size: ~2.5 GB' + #13#10 + #13#10 +
         'You will be asked to:' + #13#10 +
         '1. Configure MySQL database connection' + #13#10 +
         '2. Select a folder for batch processing',
         mbInformation, MB_OK);
  Result := True;
end;
