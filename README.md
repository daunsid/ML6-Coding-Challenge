Hello potential ML6 colleague!

If you are reading this, you are probably applying for a Machine Learning engineering job at ML6. This test will evaluate if you have the right skills for the job.
Completing the test should take about half a day if you have the relevant experience.

In this test, you will try to classify the mugs we drink from at ML6. If you are able to complete this test in a decent way, you might soon be drinking coffee from the black ML6 mugs (which is also in the data) together with us.

You can start by cloning this repository to a local and/or private location via:

```
git clone git@bitbucket.org:ml6team/challenge-find-ml6-mug.git
```

## The Data

Before you begin to implement your classification model you need to download the data used for training and local evaluation from Google Cloud Storage and place the `data` folder in the base folder. To download the data execute the following command (you will need to [install the `gsutil` command](https://cloud.google.com/storage/docs/gsutil_install#sdk-install) beforehand which is part of the *Google Cloud SDK*):

```
gsutil -m cp -R gs://ml6_junior_ml_engineer_challenge_cv_mugs_data/data .
```

For your purposes, the data has already been split into training data and evaluation data. They are respectively in the `train` folder and `eval` folder. In each you will find four folders which represent the mugs you'll need to classify. There are four kind of mugs: the white mug, the black mug (the ML6 mug), the blue mug, and the transparent mug (the glass). The white mug is class 0, the black mug class 1, the blue mug class 2, and the transparent mug class 3. These class numbers are necessary to create a correct classifier. If you want, you can inspect the data, however, the code to load the images into NumPy arrays is already written for you.


## The Model

In the `trainer` folder, you will be able to see several Python files. The `data.py`, `task.py` and `final_task.py` files are already coded for you. The only file that needs additional code is the `model.py` file. The comments in this file will indicate which code has to be written.

To test how your model is doing you can execute the following command (you will need to [install](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_sdk_version) the `gcloud` command, which is also part of the *Google Cloud SDK*):

```
gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer/ --
```

If you run this command before you wrote any code in the `model.py` file, you will notice that it returns errors. Your goal is to write code that does not return errors and achieves an accuracy that is as high as possible.

The command above will train and evaluate your classifier. The batch size and the number of epochs need to be defined in the `model.py` file.

Make sure you'll think about the solution you will submit for this coding test. If you want you can change the code written by us according to your needs. It is however important that we can still perform our automated evaluation when you submit your solution so make sure you test your solution thoroughly before you submit it. How you can test your solution is explained below.

![Data overview](data.png)

The command above uses the `task.py` file. As you can see in the figure above, this file only uses the mug images in the `train` folder and uses the images in the `eval` folder to evaluate the model. This is excellent to test how the model performs but to obtain a better evaluation one can also train upon all available data which should increase the performance on the dataset you will be evaluated on. After you finished your solution in `model.py`, you can continue reading to learn how to train your model on the full dataset.


## Deploying the Model

Once you've got the code working you will need to deploy the model to Google Cloud to turn it into an API that can receive new images of mugs and returns its prediction them. Don't worry, the code for this is already written in the `final_task.py` file. To deploy your model, you only have to run a few commands in your command line.

To export your trained model and to train your model on the images in the `train` and `eval` folder you have to execute the following command (only do this once you've completed coding the `model.py` file):

```
gcloud ai-platform local train \
    --module-name trainer.final_task \
    --package-path trainer/ --
```

Once you've executed this command, you will notice that the `output` folder was created in the root directory of this repository. This folder contains your saved model that you'll need to deploy to Google Cloud AI Platform.

In order to do so you will need to create a [Google Cloud account](https://cloud.google.com/). You will need a credit card for this, but you'll get [free credit from Google](https://cloud.google.com/free/docs/gcp-free-tier/#free-trial) to run your AI Platform instance. **Note:** if you are not eligible for the free trial please reach out to us as you are responsible for the costs associated with the project.

Once you've created your Google Cloud account, you'll need to deploy your model on a project you've created. You can follow a [Google Guide](https://cloud.google.com/ai-platform/prediction/docs/deploying-models#deploy_models_and_versions) for this. Make sure to deploy the model using Tensorflow 2.1 and we advice to use **`europe-west1` as the regional endpoint** if available. Note that the region of the bucket should correspond to the region of the model.


## Checking your Deployed Model

Before you submit your solution, you can check if your deployed model works correctly by executing the following commands:

```
MODEL_NAME=<your_model_name>
VERSION=<your_version_of_the_model>

gcloud ai-platform predict \
    --region europe-west1 \
    --model $MODEL_NAME \
    --version $VERSION \
    --json-instances check_deployed_model/test.json
```

Check if you are able to get a prediction out of the `gcloud` command. If you get errors, you should try to resolve them before submitting the solution. The output of the command should look something like this (the numbers will probably be different):

```
CLASSES  PROBABILITIES
1        [2.0589146706995187e-12, 1.0, 1.7370329621294728e-13, 1.2870057122347237e-32]
```

The values you use for the `$MODEL_NAME` variable and the `$VERSION` variable can be found in your project on the Google Cloud web interface. You will need these values and your Google Cloud *Project ID* to submit your coding test.

To be able to pass the coding test. You should be able to get an accuracy of 75% on our secret dataset of mugs (which you don't have access to). If your accuracy however seems to be less than 75% after we evaluated it, you can just keep submitting solutions until you are able to get an accuracy of 75%.


## Submitting your Coding Test

Once you are able to execute the command above without errors, you can add us to your project:

* Go to the menu of your project
* Click *IAM & admin*
* Click *Add*
* Add `ml6-coding-challenge-evaluator@recruiting-220608.iam.gserviceaccount.com`as a member with the role *Project Owner*

After you added us to your project you should fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLScW6ytY3_4yoKE39-Gd-U7WHo030YtwdggTG1D_yIQPlL7Vjg/viewform) so we are able to automatically evaluate your solution to the coding test. Once you've filled in the form you should receive an email with the results within 2 hours. We'll hope with you that your results are good enough to land an interview at ML6. If however you don't you can resubmit a new solution as many times as you want, so don't give up!

If you are invited for an interview at ML6 afterwards, make sure to bring your laptop with a copy of the code you wrote, so you can explain your `model.py` file to us.


### Taking down the deployed model

After you have received the evaluation email, we no longer require access to the model. Please check the corresponding documentation page for removing AI Platform models. This can be done via the UI or the command line.
