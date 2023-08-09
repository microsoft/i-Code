# Run on Red Hat OpenShift

This document shows how to deploy the server using Red Hat OpenShift.

## Prerequisites

You will need a running OpenShift cluster, or OKD cluster. You can provision [OpenShift on the IBM Cloud](https://cloud.ibm.com/kubernetes/catalog/openshiftcluster).

## Steps

1. [Create Watson services on IBM Cloud](#1-create-watson-services-on-ibm-cloud)
1. [Create an OpenShift project](#2-create-an-openshift-project)
1. [Create the config map](#3-create-the-config-map)
1. [Get a secure endpoint](#4-get-a-secure-endpoint)
1. [Run the web app](#5-run-the-web-app)

## 1. Create Watson services on IBM Cloud

Use the following links to create the Watson services on IBM Cloud. Copy/paste the `API Key` and `URL` or keep the browser tabs open. You'll need these later.

* [**Watson Assistant**](https://cloud.ibm.com/catalog/services/conversation)
* [**Watson Speech to Text**](https://cloud.ibm.com/catalog/services/speech-to-text)
* [**Watson Text to Speech**](https://cloud.ibm.com/catalog/services/text-to-speech)

## 2. Create an OpenShift project

* Using the OpenShift web console, select the `Application Console` view.

  ![console-options](https://raw.githubusercontent.com/IBM/pattern-utils/master/openshift/openshift-app-console-option.png)

* Use the `+Create Project` button to create a new project, then click on your project to open it.

* In the `Overview` tab, click on `Browse Catalog`.

  ![Browse Catalog](https://raw.githubusercontent.com/IBM/pattern-utils/master/openshift/openshift-browse-catalog.png)

* Choose the `Python` app container and click `Next`.

  ![Choose Node.js](https://raw.githubusercontent.com/IBM/pattern-utils/master/openshift/openshift-choose-nodejs.png)

* Give your app a name and add `https://github.com/IBM/watson-voice-bot` for the github repo, then click `Create`.

  ![Add github repo](https://raw.githubusercontent.com/IBM/pattern-utils/master/openshift/openshift-add-github-repo.png)

## 3. Create the config map

* Click on the `Resources` tab and choose `Config Maps` and then click the `Create Config Map` button.
  * Provide a `Name` for the config map.
  * Add a key named `PORT` and paste in the `8080` under `Enter a value...`.
  * Click `Add Item` and add a key named `ASSISTANT_APIKEY` and paste in the API Key from step 1 under `Enter a value...`..
  * Click `Add Item` and add a key named `ASSISTANT_URL` and paste in the URL from step 1 under `Enter a value...`..
  * Click `Add Item` and add a key named `SPEECH_TO_TEXT_APIKEY` and paste in the API Key from step 1 under `Enter a value...`..
  * Click `Add Item` and add a key named `SPEECH_TO_TEXT_URL` and paste in the URL from step 1 under `Enter a value...`..
  * Click `Add Item` and add a key named `TEXT_TO_SPEECH_APIKEY` and paste in the API Key from step 1 under `Enter a value...`..
  * Click `Add Item` and add a key named `TEXT_TO_SPEECH_URL` and paste in the URL from step 1 under `Enter a value...`..
  * Hit the `Create` button.
  * Click on your new Config Map's name.
  * Click the `Add to Application` button.
  * Select your application from the pulldown.
  * Click `Save`.

  ![config_map.png](images/config_map.png)

* Go to the `Applications` tab, choose `Deployments` to view the status of your application.

## 4. Get a secure endpoint

* From the OpenShift or OKD UI, under `Applications` ▷ `Routes` you will see your app.
  * Click on the application `Name`.
  * Under `TLS Settings`, click on `Edit`.
  * Under `Security`, check the box for `Secure route`.
  * Hit `Save`.

## 5. Run the web app

* Go back to `Applications` ▷ `Routes`. You will see your app.
* Click your app's `Hostname`. This will open the Watson Voice Bot web app in your browser.
* Go back to the README to see an example using the default insurance bot dialog.

[![return](https://raw.githubusercontent.com/IBM/pattern-utils/master/deploy-buttons/return.png)](https://github.com/IBM/watson-voice-bot#sample-output)
