; VideoHighlighter setup with PyTorch as a pack (Inno Setup 6.1+).
;
; A real installer: Inno compresses and installs the files itself, so there is
; no .7z inside and nothing to download before the app works. What it carries:
;
;   * the core app (PyInstaller's dist/VideoHighlighter, built without torch);
;   * the CPU PyTorch pack, so every machine has a working app offline.
;
; What it can fetch, each a checkbox, downloaded from PackBase during setup
; and verified against its SHA-256:
;
;   * the NVIDIA pack (torch built for CUDA, ~4.4 GB installed) - ticked when
;     an NVIDIA card is found. It replaces the CPU pack only once it is fully
;     unpacked. How packs load: packaging/packs/rthook_packs.py.
;   * the CLIP model (visual search, chapters, taught categories) - ticked by
;     default. Model weights barely compress, so carrying it would put its
;     whole size into every download, including for people who never search.
;     It lands in {app}\models\, the first place llm/clip_prefilter.py looks.
;   * the multilingual search text model (llm/clip_multilingual.py) - ticked
;     when the language chosen is not English. Only offered when this Setup
;     was built against packs that have it (MlPackAsset).
;
; Any of them failing or being declined still leaves an app that runs.
;
; ALREADY INSTALLED. Run over an existing install, Setup finds the packs it
; already has (here, or where the app put them in the per-user folder) and
; leaves them unticked, marked "already installed", instead of downloading
; them again. The NVIDIA pack counts only when it is the version this Setup
; was built against (CudaPackVersion).
;
; THE LANGUAGE. Setup opens by asking its own language, preselecting Windows'
; display language, and every page after that is in it: Inno Setup's standard
; texts from its translations (official ones from the compiler,
; ChineseSimplified, Indonesian and Vietnamese vendored in languages/ from
; jrsoftware/issrc at is-6_7_1, the compiler CI pins), this Setup's own from
; setup-messages.iss. The same choice is the app's language: Setup leaves it
; in {app}\install_language.txt, which the app reads at its first start
; (modules/i18n/content_language.py) - its menus, and the language it listens
; for in transcripts, writes subtitles and narration in, and understands in
; search. Not written over an install that already has a config: the app
; holds that choice already.
;
; Built in CI:
;   ISCC.exe /DAppVersion= /DDistDir= /DCpuPackDir= /DPackBase= \
;            /DCudaPackAsset= /DCudaPackSha256= /DCudaPackVersion= /DCudaPackMB= \
;            /DCudaInstalledMB= /DInstalledMB= packaging\installer\videohighlighter-packs.iss
;
; FOR ME ONLY, OR FOR ALL USERS. Setup asks. "For me only" installs under
; %LOCALAPPDATA%\Programs with no admin prompt, and is the one the in-app
; updater can update in place. "For all users" installs under Program Files
; after one UAC prompt; the app runs the same (app_paths.user_data_dir moves
; its data to the user's profile when its own folder is read-only), but a
; normal user cannot write there, so updates for such an install come as a
; download (update_check.install_dir_writable) rather than in place.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef AppName
  #define AppName "VideoHighlighter"
#endif
#ifndef AppId
  #define AppId "{6E1B9C34-4F27-4A88-9D0E-1C5A7B3F8E42}"
#endif
#ifndef OutputName
  #define OutputName "VideoHighlighter-Setup"
#endif
#ifndef DistDir
  #define DistDir "..\..\dist\VideoHighlighter"
#endif
#ifndef CpuPackDir
  #define CpuPackDir "..\..\build-packs\torch-cpu"
#endif
; Where the NVIDIA pack is downloaded from, and what it must hash to.
#ifndef PackBase
  #define PackBase ""
#endif
; The download mirror's copy of the same packs tag (tools/mirror_release.py),
; tried before PackBase. Same files, checked against the same SHA-256.
#ifndef PackMirror
  #define PackMirror ""
#endif
#ifndef CudaPackAsset
  #define CudaPackAsset "component-torch-cu128.7z"
#endif
#ifndef CudaPackSha256
  #define CudaPackSha256 ""
#endif
; The NVIDIA pack's version (its pack.json), to tell an installed copy of this
; pack from an older one. Empty: any installed NVIDIA pack counts.
#ifndef CudaPackVersion
  #define CudaPackVersion ""
#endif
#ifndef CudaPackMB
  #define CudaPackMB 0
#endif
#ifndef CudaInstalledMB
  #define CudaInstalledMB 0
#endif
#ifndef ClipPackAsset
  #define ClipPackAsset "component-models-clip.7z"
#endif
#ifndef ClipPackSha256
  #define ClipPackSha256 ""
#endif
#ifndef ClipPackMB
  #define ClipPackMB 0
#endif
#ifndef ClipInstalledMB
  #define ClipInstalledMB 0
#endif
#ifndef MlPackAsset
  #define MlPackAsset ""
#endif
#ifndef MlPackSha256
  #define MlPackSha256 ""
#endif
#ifndef MlPackMB
  #define MlPackMB 0
#endif
#ifndef MlInstalledMB
  #define MlInstalledMB 0
#endif
#ifndef InstalledMB
  #define InstalledMB 0
#endif

#define Publisher "Aseiel"
#define AppExe "VideoHighlighter.exe"
; Held by the running app (modules/system/app_mutex.py). While it exists,
; Setup and the uninstaller ask the user to close the app instead of
; deleting around the files it has open. Named after AppName, so the Pro
; installer waits only for Pro.
#define AppMutex AppName + " is running"
#define CudaPackName "torch-cu128"
#define ClipDirName "clip-vit-base-patch32-ov"
#define MlDirName "clip-vit-b32-multilingual"

[Setup]
AppId={{#AppId}
AppName={#AppName}
AppMutex={#AppMutex},Global\{#AppMutex}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; {autopf} follows the choice: Program Files for all users, the per-user
; %LOCALAPPDATA%\Programs for me only.
DefaultDirName={autopf}\{#AppName}
UsedUserAreasWarning=no
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
OutputBaseFilename={#OutputName}
SetupIconFile=..\..\assets\icon.ico
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Asked first, preselecting Windows' display language; also the app's language.
ShowLanguageDialog=yes

; The app's interface languages (modules/i18n/languages.py). The names are
; what ChosenLanguage turns back into the app's codes.
[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "zhCN"; MessagesFile: "languages\ChineseSimplified.isl"
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "ptBR"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "ja"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"
Name: "ko"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "id"; MessagesFile: "languages\Indonesian.isl"
Name: "tr"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "pl"; MessagesFile: "compiler:Languages\Polish.isl"
Name: "vi"; MessagesFile: "languages\Vietnamese.isl"
Name: "nl"; MessagesFile: "compiler:Languages\Dutch.isl"

#include "setup-messages.iss"

; Their order is the tasks list's: DescribeTasks addresses them by index.
[Tasks]
Name: "nvidia"; Description: "{cm:NvidiaTask,{#CudaPackMB}}"; Flags: unchecked
Name: "clip"; Description: "{cm:ClipTask,{#ClipPackMB}}"
#if MlPackAsset != ""
Name: "multilingual"; Description: "{cm:MlTask,{#MlPackMB}}"; Flags: unchecked
#endif
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"

; The bundle folder is replaced whole on every install. A build from before
; packs carried torch inside _internal, and _internal is on the frozen app's
; import path ahead of any pack: left behind, that old torch would shadow the
; pack. User data never lives there (app_paths.user_data_dir is the exe's
; folder), so nothing of the user's goes with it.
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "7zr.exe"; Flags: dontcopy
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Skipped when the NVIDIA pack is already installed (Setup run again over it).
Source: "{#CpuPackDir}\*"; DestDir: "{app}\packs\torch-cpu"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: NeedCpuPack

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

; Packs unpacked by 7-Zip and everything the app wrote at runtime are not
; tracked by the installer, so the folder is removed whole — the same full
; removal videohighlighter.iss promises.
[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
var
  DownloadPage: TDownloadWizardPage;
  NvidiaFound: Boolean;
  NvidiaName: String;
  NvidiaDefaultApplied: Boolean;
  CudaPackDownloaded: Boolean;
  ClipPackDownloaded: Boolean;
  MlPackDownloaded: Boolean;
  MlDefaultApplied: Boolean;
  ClipDefaultApplied: Boolean;

// The language Setup runs in, as the app's code for it: the [Languages] names
// are the app's codes without their hyphen.
function ChosenLanguage: String;
begin
  Result := ActiveLanguage;
  if Result = 'zhCN' then
    Result := 'zh-CN'
  else if Result = 'ptBR' then
    Result := 'pt-BR';
end;

// An install that already has a config: the app holds the language already,
// beside the exe or, for a read-only install, in the per-user folder.
function HasConfig: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\config.yaml'))
    or FileExists(ExpandConstant('{localappdata}\{#AppName}\config.yaml'));
end;

// True when Windows reports an NVIDIA display adapter. Any failure (WMI
// disabled, a locked-down machine) is "not found": the checkbox then simply
// starts unticked, and the user can still tick it.
function HasNvidiaGpu: Boolean;
var
  Locator, Service, Items, Item: Variant;
  I: Integer;
  AdapterName: String;
begin
  Result := False;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Items := Service.ExecQuery('SELECT Name FROM Win32_VideoController');
    for I := 0 to Items.Count - 1 do
    begin
      Item := Items.ItemIndex(I);
      // WMI hands back a Variant; string functions need a String.
      AdapterName := Item.Name;
      Log('Display adapter: ' + AdapterName);
      if Pos('NVIDIA', Uppercase(AdapterName)) > 0 then
      begin
        Result := True;
        NvidiaName := AdapterName;
      end;
    end;
  except
    Log('GPU detection failed: ' + GetExceptionMessage);
  end;
end;

function NeedCpuPack: Boolean;
begin
  Result := not DirExists(ExpandConstant('{app}\packs\{#CudaPackName}'));
end;

// ---------------------------------------------------------------------------
// What an earlier install already has. The app downloads a pack beside itself
// when it can write there, else into the per-user folder
// (modules/packs/pack_manager.target_root), so both are looked in.
// ---------------------------------------------------------------------------

function PackRoots(I: Integer): String;
begin
  if I = 0 then
    Result := ExpandConstant('{app}')
  else
    Result := ExpandConstant('{localappdata}\{#AppName}');
end;

// This Setup's NVIDIA pack, unpacked. An older version of it (another torch)
// does not count: Setup should offer the new one.
function CudaInstalled: Boolean;
var
  I: Integer;
  Dir: String;
  Info: AnsiString;
begin
  Result := False;
  for I := 0 to 1 do
  begin
    Dir := PackRoots(I) + '\packs\{#CudaPackName}';
    if DirExists(Dir + '\site-packages')
       and LoadStringFromFile(Dir + '\pack.json', Info)
       and (('{#CudaPackVersion}' = '') or (Pos('{#CudaPackVersion}', String(Info)) > 0)) then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

// A model pack's folder under models\, with the file that proves it is there
// (pack_manager.MODEL_DIRS has the same pairs).
function ModelInstalled(const DirName, Probe: String): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 0 to 1 do
    if FileExists(PackRoots(I) + '\models\' + DirName + '\' + Probe) then
    begin
      Result := True;
      Exit;
    end;
end;

function ClipInstalled: Boolean;
begin
  Result := ModelInstalled('{#ClipDirName}', 'openvino_model.xml');
end;

function MlInstalled: Boolean;
begin
  Result := ModelInstalled('{#MlDirName}', 'text.onnx');
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if ProgressMax > 0 then
    // No line of [Code] may start with "[": Inno reads it as a section.
    DownloadPage.SetText(FileName, FmtMessage(CustomMessage('DownloadedOf'), [
      IntToStr(Progress div 1048576), IntToStr(ProgressMax div 1048576)]));
  Result := True;
end;

procedure InitializeWizard;
begin
  NvidiaFound := HasNvidiaGpu;
  DownloadPage := CreateDownloadPage(CustomMessage('DownloadingTitle'),
    CustomMessage('DownloadingDescription'), @OnDownloadProgress);
end;

procedure AppendToTask(Index: Integer; const Note: String);
begin
  if Index < WizardForm.TasksList.Items.Count then
    WizardForm.TasksList.ItemCaption[Index] :=
      WizardForm.TasksList.ItemCaption[Index] + ' - ' + Note;
end;

// Say what was found, so each checkbox explains itself (and a tester can tell
// detection from a wrong default at a glance). By index, in [Tasks] order,
// because the captions are in whatever language Setup runs in.
procedure DescribeTasks;
begin
  if CudaInstalled then
    AppendToTask(0, CustomMessage('AlreadyInstalled'))
  else if NvidiaFound then
    AppendToTask(0, FmtMessage(CustomMessage('NvidiaRecommended'), [NvidiaName]))
  else
    AppendToTask(0, CustomMessage('NvidiaNotFound'));
  if ClipInstalled then
    AppendToTask(1, CustomMessage('AlreadyInstalled'));
#if MlPackAsset != ""
  if MlInstalled then
    AppendToTask(2, CustomMessage('AlreadyInstalled'));
#endif
end;

// The defaults, set once, the first time the tasks page shows - after that
// the user's own choice stands, including on Back/Next. A pack already
// installed starts unticked; ticking it again downloads it again.
procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpSelectTasks) and not NvidiaDefaultApplied then
  begin
    DescribeTasks;
    if NvidiaFound and not CudaInstalled then
      WizardSelectTasks('nvidia');
    NvidiaDefaultApplied := True;
  end;
  if (CurPageID = wpSelectTasks) and not ClipDefaultApplied then
  begin
    if ClipInstalled then
      WizardSelectTasks('!clip');
    ClipDefaultApplied := True;
  end;
#if MlPackAsset != ""
  // Ticked for any language but English when it is not there yet.
  if (CurPageID = wpSelectTasks) and not MlDefaultApplied then
  begin
    if (not HasConfig) and (ChosenLanguage <> 'en') and not MlInstalled then
      WizardSelectTasks('multilingual');
    MlDefaultApplied := True;
  end;
#endif
end;

// Whether the multilingual task exists in this build and is ticked.
function MlSelected: Boolean;
begin
#if MlPackAsset != ""
  Result := WizardIsTaskSelected('multilingual');
#else
  Result := False;
#endif
end;

function FreeSpaceMB(const Path: String): Int64;
var
  Free, Total: Int64;
begin
  if GetSpaceOnDisk64(AddBackslash(ExtractFileDrive(Path)), Free, Total) then
    Result := Free div 1048576
  else
    Result := -1;
end;

// The install plus, with the NVIDIA pack, its download in {tmp} and its
// unpacked size. Checked before anything is fetched, so a full disk is a
// message naming both numbers instead of a failure at the end of 1.5 GB.
function SpaceLooksSufficient: Boolean;
var
  Need, Free: Int64;
  AppDir: String;
begin
  Result := True;
  Need := {#InstalledMB};
  if WizardIsTaskSelected('nvidia') then
    Need := Need + {#CudaPackMB} + {#CudaInstalledMB};
  if WizardIsTaskSelected('clip') then
    Need := Need + {#ClipPackMB} + {#ClipInstalledMB};
  if MlSelected then
    Need := Need + {#MlPackMB} + {#MlInstalledMB};
  if Need <= 0 then
    Exit;
  AppDir := ExpandConstant('{app}');
  Free := FreeSpaceMB(AppDir);
  if (Free >= 0) and (Free < Need) then
  begin
    MsgBox(FmtMessage(CustomMessage('LowSpace'), [
      ExtractFileDrive(AppDir), IntToStr(Need), IntToStr(Free)]), mbError, MB_OK);
    Result := False;
  end;
end;

// One attempt at one host. Aborted is set when the user pressed Cancel, and
// Error to what went wrong otherwise.
function DownloadFrom(const Url, Asset, Sha256: String; var Aborted: Boolean; var Error: String): Boolean;
begin
  Result := False;
  DownloadPage.Clear;
  DownloadPage.Add(Url, Asset, Sha256);
  try
    DownloadPage.Download;
    Result := True;
  except
    Aborted := DownloadPage.AbortedByUser;
    Error := GetExceptionMessage;
  end;
end;

// Fetch one optional component into {tmp}: from the mirror when this Setup
// has one, then from PackBase. True when it arrived and matched its SHA-256.
// On failure the user decides: Yes installs without it, No sets Stay, which
// keeps the wizard on the Ready page so Next retries.
function FetchComponent(const Asset, Sha256, What, Without: String; var Stay: Boolean): Boolean;
var
  Aborted: Boolean;
  Error: String;
begin
  Aborted := False;
  Error := '';
  Result := False;
#if PackMirror != ""
  Result := DownloadFrom('{#PackMirror}/' + Asset, Asset, Sha256, Aborted, Error);
  if not Result and not Aborted then
    Log(What + ': the mirror failed (' + Error + '); trying {#PackBase}');
#endif
  if not Result and not Aborted then
    Result := DownloadFrom('{#PackBase}/' + Asset, Asset, Sha256, Aborted, Error);
  if Result then
    Exit;
  if Aborted then
    Log(What + ': download cancelled by the user.')
  else
    Log(What + ': download failed: ' + Error);
  Stay := SuppressibleMsgBox(
    FmtMessage(CustomMessage('CouldNotDownload'), [What, AddPeriod(Error), Without]),
    mbConfirmation, MB_YESNO, IDYES) <> IDYES;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Stay: Boolean;
begin
  Result := True;
  if CurPageID <> wpReady then
    Exit;

  if not SpaceLooksSufficient then
  begin
    Result := False;
    Exit;
  end;

  CudaPackDownloaded := False;
  ClipPackDownloaded := False;
  MlPackDownloaded := False;
  if not (WizardIsTaskSelected('nvidia') or WizardIsTaskSelected('clip') or MlSelected) then
    Exit;
  if '{#PackBase}' = '' then
  begin
    Log('No PackBase compiled in; nothing optional is downloaded.');
    Exit;
  end;

  Stay := False;
  DownloadPage.Show;
  try
    if WizardIsTaskSelected('nvidia') then
      CudaPackDownloaded := FetchComponent('{#CudaPackAsset}', '{#CudaPackSha256}',
        CustomMessage('NvidiaName'), CustomMessage('NvidiaWithout'), Stay);
    if (not Stay) and WizardIsTaskSelected('clip') then
      ClipPackDownloaded := FetchComponent('{#ClipPackAsset}', '{#ClipPackSha256}',
        CustomMessage('ClipName'), CustomMessage('ClipWithout'), Stay);
#if MlPackAsset != ""
    if (not Stay) and MlSelected then
      MlPackDownloaded := FetchComponent('{#MlPackAsset}', '{#MlPackSha256}',
        CustomMessage('MlName'), CustomMessage('MlWithout'), Stay);
#endif
  finally
    DownloadPage.Hide;
  end;
  Result := not Stay;
end;

// 7-Zip an archive from {tmp} into Dest. False (with the exit code logged)
// when it could not; Dest is then removed again.
function Unpack(const Asset, Dest: String): Boolean;
var
  ResultCode: Integer;
begin
  ExtractTemporaryFile('7zr.exe');
  DelTree(Dest, True, True, True);
  Result := Exec(ExpandConstant('{tmp}\7zr.exe'),
                 Format('x -y "%s" -o"%s"', [ExpandConstant('{tmp}\') + Asset, Dest]),
                 '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  if not Result then
  begin
    Log('7-Zip failed on ' + Asset + ' (exit code ' + IntToStr(ResultCode) + ')');
    DelTree(Dest, True, True, True);
  end;
end;

// Unpack the NVIDIA pack beside the CPU one, and only when that worked, remove
// the CPU pack. Until the last step the CPU pack is untouched, so any failure
// here leaves an install that starts.
procedure InstallCudaPack;
var
  Target, Staging: String;
begin
  Target := ExpandConstant('{app}\packs\{#CudaPackName}');
  Staging := Target + '.partial';
  WizardForm.StatusLabel.Caption := CustomMessage('UnpackingNvidia');
  WizardForm.Refresh;

  if Unpack('{#CudaPackAsset}', Staging) then
  begin
    DelTree(Target, True, True, True);
    if RenameFile(Staging, Target) then
    begin
      DelTree(ExpandConstant('{app}\packs\torch-cpu'), True, True, True);
      Exit;
    end;
    DelTree(Staging, True, True, True);
  end;
  MsgBox(CustomMessage('NvidiaFailed'), mbError, MB_OK);
end;

// A model archive holds models\<DirName>\...: unpack it to a staging folder
// and move that one directory into {app}\models. Unpacking and Failed are
// what the user sees meanwhile and if it fails.
procedure InstallModelPack(const Asset, DirName, Unpacking, Failed: String);
var
  Staging, Target: String;
begin
  Staging := ExpandConstant('{app}\models\.') + DirName + '.partial';
  Target := ExpandConstant('{app}\models\') + DirName;
  WizardForm.StatusLabel.Caption := Unpacking;
  WizardForm.Refresh;

  ForceDirectories(ExpandConstant('{app}\models'));
  if Unpack(Asset, Staging) then
  begin
    DelTree(Target, True, True, True);
    if RenameFile(Staging + '\models\' + DirName, Target) then
    begin
      DelTree(Staging, True, True, True);
      Exit;
    end;
    DelTree(Staging, True, True, True);
  end;
  MsgBox(Failed, mbError, MB_OK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;
  if CudaPackDownloaded then
    InstallCudaPack;
  if ClipPackDownloaded then
    InstallModelPack('{#ClipPackAsset}', '{#ClipDirName}',
      CustomMessage('UnpackingClip'), CustomMessage('ClipFailed'));
#if MlPackAsset != ""
  if MlPackDownloaded then
    InstallModelPack('{#MlPackAsset}', '{#MlDirName}',
      CustomMessage('UnpackingMl'), CustomMessage('MlFailed'));
#endif
  // Read by the app at its first start; ignored once its config has a language.
  if not HasConfig then
    SaveStringToFile(ExpandConstant('{app}\install_language.txt'), ChosenLanguage, False);
end;
