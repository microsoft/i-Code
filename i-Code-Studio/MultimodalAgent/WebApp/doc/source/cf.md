# Run on IBM Cloud with Cloud Foundry

This document shows how to deploy the server using Cloud Foundry on IBM Cloud.

[![Deploy to IBM Cloud](https://cloud.ibm.com/devops/setup/deploy/button_x2.png)](https://cloud.ibm.com/devops/setup/deploy?repository=https://github.com/IBM/watson-voice-bot.git)

1. Click the above `Deploy to IBM Cloud` button and then click on the `Delivery Pipeline` tool integration.

   ![deploy](images/cf_deploy.png)

2. Create an API key by pressing the `Create+` button located next to the `IBM Cloud API key` field and then `Create` in the pop-up.

3. Select your `Region`, `Organization` and `Space`.

4. Click `Create` at the top of the panel to start the deployment process.

5. From the Toolchains view, click on the `Delivery Pipeline` to watch while the app is deployed. Here you'll be able to see logs about the deployment.

   ![toolchain_pipeline](images/toolchain_pipeline.png)

6. To see the app and services created and configured for this code pattern, use the [IBM Cloud](https://cloud.ibm.com) dashboard. The app is named `watson-voice-bot` with a unique suffix. The following services are created and easily identified by the `wvb-` prefix:

   * wvb-watson-assistant
   * wvb-text-to-speech
   * wvb-speech-to-text

7. Click on the app and then click on `Visit App URL` visit the bot's web page.
8. Click on the microphone icon to begin speaking and click it again when you are finished.

[![return](https://raw.githubusercontent.com/IBM/pattern-utils/master/deploy-buttons/return.png)](https://github.com/IBM/watson-voice-bot#sample-output)
