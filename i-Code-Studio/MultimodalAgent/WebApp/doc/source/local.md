# Run locally

This document shows how to run the `watson-voice-bot` server on your local machine.

## Steps

1. [Clone the repo](#1-clone-the-repo)
2. [Create Watson services on IBM Cloud](#2-create-watson-services-on-ibm-cloud)
3. [Upload the Watson Assistant workspace](#3-upload-the-watson-assistant-workspace)
4. [Configure `.env` with credentials](#4-configure-env-with-credentials)
5. [Run the application](#5-run-the-application)

## 1. Clone the repo

Use the following command to clone the watson-voice-bot GitHub repository.

```bash
git clone https://github.com/IBM/watson-voice-bot
```

## 2. Create Watson services on IBM Cloud

Use the following links to create the Watson services on IBM Cloud:

* [**Watson Assistant**](https://cloud.ibm.com/catalog/services/conversation)
* [**Watson Speech to Text**](https://cloud.ibm.com/catalog/services/speech-to-text)
* [**Watson Text to Speech**](https://cloud.ibm.com/catalog/services/text-to-speech)

## 3. Upload the Watson Assistant workspace

* Find the Assistant service in your IBM Cloud Dashboard `Services`.
* Click on your Watson Assistant service and then click on `Launch Watson Assistant`.
* Use the left sidebar and click on the `Skills` icon.
* Click the `Create skill` button.
* Select the `Dialog skill` card and click `Next`.
* Select the `Import skill` tab.
* Click the `Choose JSON File` button and choose the `data/skill-insurance-voice-bot.json` file in your cloned watson-voice-bot repo.
* Make sure the `Everything` button is enabled.
* Click `Import`.
* Go back to the Skills page (use the left sidebar).
* Look for `insurance-voice-bot` card.
* Click on the three dots in the upper right-hand corner of the card and select `View API Details`.
* Copy the `Workspace ID` GUID. Save it for the next step.
  !["Get Workspace ID"](https://raw.githubusercontent.com/IBM/pattern-utils/master/watson-assistant/assistantPostSkillGetID.gif)

## 4. Configure `.env` with credentials

Our services are created and the workspace is uploaded. It's now time to let our application run locally and to do that we'll configure a simple text file with the environment variables we want to use. We begin by copying the [`sample.env`](sample.env) file and naming it `.env`.

```bash
cp sample.env .env
```

First, edit the .env file and set the `WORKSPACE_ID` to the value that was retrieved in the previous step.

Next, set the key-value pairs with credentials for each IBM Cloud service (Assistant, Speech to Text, and Text to Speech).

* Find each service in your IBM Cloud Dashboard `Services`.
* Click on a service to view its `Manage` page.
* Use the copy icon and copy/paste the `API Key` and `URL` into your .env file for each service.

### `sample.env`

```bash
# Copy this file to .env before starting the app.
# Replace the credentials with your own.

# Watson Speech to Text
SPEECH_TO_TEXT_APIKEY=<add_speech-to-text_apikey>
SPEECH_TO_TEXT_URL=<add_speech-to-text_url>

# Watson Text to Speech
TEXT_TO_SPEECH_APIKEY=<add_text-to-speech_apikey>
TEXT_TO_SPEECH_URL=<add_text-to-speech_url>

# Watson Assistant
ASSISTANT_APIKEY=<add_assistant_apikey>
ASSISTANT_URL=<add_assistant_url>

# Optionally, use a non-default skill by specifying your own workspace ID or name.
# WORKSPACE_ID=<add_assistant_workspace_id>
# WORKSPACE_NAME=<add_assistant_workspace_name>
```

## 5. Run the application

* The server requires Python 3.5 and above. It has been tested with Python 3.8.0.

* The general recommendation for Python development is to use a virtual environment ([venv](https://docs.python.org/3/tutorial/venv.html)). To install and initialize a virtual environment, use the `venv` module on Python 3:

Create the virtual environment using Python. Use one of the two commands depending on your Python version.
> Note: `python` may be named `python3` on your system.

```bash
python -m venv mytestenv
```

Now source the virtual environment. Use one of the two commands depending on your OS.

```bash
source mytestenv/bin/activate  # Mac or Linux
./mytestenv/Scripts/activate   # Windows PowerShell
```

> **TIP** :bulb: To terminate the virtual environment use the `deactivate` command.

1. Start the app by running:

  ```bash
  pip install -r requirements.txt
  python app.py
  ```

1. Launch a browser and navigate to [http://localhost:5000](http://localhost:5000)
1. Click on the microphone icon to begin speaking and click it again when you are finished.

[![return](https://raw.githubusercontent.com/IBM/pattern-utils/master/deploy-buttons/return.png)](https://github.com/IBM/watson-voice-bot#sample-output)
