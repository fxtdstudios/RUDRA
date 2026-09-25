; The RUDRA installer for Windows x64 (Inno Setup 6). scripts/PACKAGE_WINDOWS.ps1
; compiles it from the packaged folder:
;   ISCC /DAppVersion=0.9.0-beta.1 /DSourceDir=<dist\beta\RUDRA-...-windows-x64> /DOutputDir=<dist\beta> rudra.iss
; Per-user by default (no administrator prompt); "Install for all users" is
; offered and then needs one.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #error SourceDir (the packaged folder) is required
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif

[Setup]
AppId={{6F1E2B7A-3C4D-4E5F-9A8B-52554452410A}
AppName=RUDRA
AppVersion={#AppVersion}
AppVerName=RUDRA {#AppVersion}
AppPublisher=FXTD Studios
AppPublisherURL=https://fxtdstudios.com
AppSupportURL=https://github.com/fxtdstudios/RUDRA/issues
AppCopyright=Copyright 2026 FXTD Studios. PolyForm Noncommercial 1.0.0.
DefaultDirName={autopf}\RUDRA
DefaultGroupName=RUDRA
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile={#SourceDir}\LICENSE
SetupIconFile=..\icons\rudra.ico
UninstallDisplayIcon={app}\RUDRA.exe
UninstallDisplayName=RUDRA {#AppVersion}
OutputDir={#OutputDir}
OutputBaseFilename=RUDRA-{#AppVersion}-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\RUDRA"; Filename: "{app}\RUDRA.exe"
Name: "{autodesktop}\RUDRA"; Filename: "{app}\RUDRA.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\RUDRA.exe"; Description: "{cm:LaunchProgram,RUDRA}"; Flags: nowait postinstall skipifsilent
