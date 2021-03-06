![picture](https://github.com/samho22/QTM350/blob/539e01e2544560a771c3265c088b334d2c5d13fa/photos/Screen%20Shot%202021-04-22%20at%202.28.12%20PM.png)

# **Analyzing the Accuracy of Amazon Translate Service using Different Languages**



Hello! Welcome to Tricolore's Final Project repository. In this repo, we will be working with Amazon Translate, which is an AWS ML service that converts between languages. Amazon Translate is based on neural networks trained for language translation. This program allows users to translate between a given source language and a target language by inputing a source text and receiving an output text in the target language.

This readme will walk you through how to navigate our repo. There are two ways you can use Translate to replicate our project: in SDK Python and using the Amazon Translate service directly. 

Here is a link to our blog webpage: [TricoloreFinalProjectBlog](https://qtm350finalprojectblog.s3.amazonaws.com/TricoloreProjectBlogFinal.html)

## **Option 1: Working with the Translate API ML service in SDK Python**
**Step 1: Setup**



Setting up an IAM role 
In order to use this API within Sagemaker, we will need to update the Role we have been using to control Sagemaker permissions. Recall, when you created your Sagemaker instance, one of the steps was creating a new IAM Role. If you used the suggested default, the name would be similar to AmazonSageMaker-ExecutionRole-0238127377.

Under the heading "Permissions and encryption" in your desired notebook instance, click the link to the IAM role ARN.

![picture](https://github.com/samho22/QTM350/blob/8da5b18bfdae27720d58d5519f3d6dce6ed3e705/photos/Screen%20Shot%202021-04-22%20at%202.28.22%20PM.png)

### Adding policies
To use the examples we will present for working with Amazong Translate, you will need to add TranslateFullAccess permissions. This permission is required to work with the AWS translate function. 

To add it, in the IAM role Summary page (pictured in the screenshot below), click the blue "Attach policies" button. In the search bar, type TranslateFullAccess, select it by ticking the empty white box next to the name when it appears, and then click the blue "Attach policy" button.

![picture](https://github.com/samho22/QTM350/blob/505d792808ac4436b05ea1ecc0f14f4f40d6a609/photos/Screen%20Shot%202021-04-22%20at%203.22.52%20PM.png)


To use the translate service in practical situations within buisness or personal needs, we will be writing python code and using the AWS service given to us. Therefore, we need to import boto3 in order to integrate your Python application, library, or script with AWS services (the translate service we will be using). Also, import json so we store the translated text into json files.


```python
import boto3
import json
```

**Step 2: Storing the text you want to translate**

First, we need to create a dictionary with three kets: Text, SourceLanguageCode, and TargetLanguageCode. The Text key will store the text that you wish to translate. The SourceLanguageCode will store the language of the original text. TargetLanguageCode will store the language you wish to translate the text to.

AWS translate offers many languages. You can find the language codes [here](https://docs.aws.amazon.com/translate/latest/dg/what-is.html).



The following are the list of phrases that we used for our translations. There are two pairs of translations. As you can see, the formal phrase and the slang phrase for each grouping convey the same message, just using different words. 

*  **Formal Phrase 1**: Please accept my sincere apology for the mixup about your Starbucks order. I will gladly make you a new drink. 
*  **Slang Phrase 1**: Hey, my bad for messing up your Starbucks drink, so sorry. I can make you a new one if you want? 
*  **Formal Phrase 2**: Hello! I wanted to confirm that you all were still available to meet tonight? It will be a casual event, however, I am in need of transportation to the function.

*  **Slang Phrase 2**: Yo! Are you guys still down to chill tonight? It’s gonna be super low key, but I kinda need a ride to the party.

In this example, we will be translating "Please accept my sincere apology for the mixup about your Starbucks order. I will gladly make you a new drink." from English to Korean.


First, we will store the English phrase, the target language code (ko), and the source language code (en) in a dictionary.


```python
Amazon_Translate = {
    "Text": "Please accept my sincere apology for the mixup about your Starbucks order. I will gladly make you a new drink.", 
    "SourceLanguageCode": "en", 
    "TargetLanguageCode": "ko"
}
```

Now, we are going to save this dictionary as a json file.




```python
with open('Amazon_Translate.json', 'w') as fp:
    json.dump(Amazon_Translate, fp)
```

![picture](https://github.com/samho22/QTM350/blob/505d792808ac4436b05ea1ecc0f14f4f40d6a609/photos/Screen%20Shot%202021-04-22%20at%203.23.16%20PM.png)

Using boto3, we can call the Translate API. We are using the translate-text function in order to translate the text. This will save a new json file with the translated text. I suggest naming the new json file as yourfilename_translated, so you know it is the new json with the translations.




```python
!aws translate translate-text \
            --region us-east-1 \
            --cli-input-json file://Amazon_Translate.json > Amazon_Translate_Translated.json
```

![picture](https://github.com/samho22/QTM350/blob/505d792808ac4436b05ea1ecc0f14f4f40d6a609/photos/Screen%20Shot%202021-04-22%20at%203.23.23%20PM.png)


If you want to the store the json files you've made after translating into a bucket, this code is an easy way to store them. The bucket name should be an existing bucket. Option 2 will go more into depth about how to create a bucket. 


```python
s3 = boto3.client('s3')
filename = 'Amazon_Translate_Translated.json'
bucket_name = 'aws-translations'
s3.upload_file(filename, bucket_name, filename)
```

## Option 2: Using the AWS Batch Translate service
*This is the option we used to create the translations for our project.*

**Step One: Setting up your buckets**

#### First Bucket: Phrase Inputs
To translate these four phrases in a more efficient manner, we used the batch translate feature in the Amazon Translate service. First, to use this service, we need to store all of the phrases as a text file in an S3 Bucket. S3 is another service in AWS that you can learn about on the [AWS website. ](https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html)

Log into your AWS account and go to the S3 service. From there, create a new bucket for this task. To do this, click Create Bucket in the top right-hand corner and choose the permissions that you want. This bucket will contain all of the phrases, texts, documents, etc. that you would like to translate. 


![picture](https://github.com/samho22/QTM350/blob/505d792808ac4436b05ea1ecc0f14f4f40d6a609/photos/Screen%20Shot%202021-04-22%20at%203.23.37%20PM.png)

For batch translation service to work, you must create a new folder in the bucket. To do this, click "Create folder" in the top right-hand corner. We created a new folder called "Phrases".

![picture](https://github.com/samho22/QTM350/blob/505d792808ac4436b05ea1ecc0f14f4f40d6a609/photos/Screen%20Shot%202021-04-22%20at%203.23.44%20PM.png)

In this folder, drag and drop your files with the phrases you would like to translate. Keep in mind that all of the files within the folder should be the same format. You should find all of the files you have uploaded where it says "Objects". We uploaded four text files with the four phrases that we would like to see translated.

![picture](https://github.com/samho22/QTM350/blob/505d792808ac4436b05ea1ecc0f14f4f40d6a609/photos/Screen%20Shot%202021-04-22%20at%203.23.52%20PM.png)

#### Second Bucket: Phrase Outputs
Now that we have a bucket with everything we would like to translate, we need to create a bucket to store all of the translations. Create a new bucket and choose the permissions that you want. Again, you must create a new folder in the bucket. In our case, because we would like all of the phrases to be translated into eight different languages, we created eight different folders. This bucket will contain all of the translated phrases, texts, documents, etc.


![picture](https://github.com/samho22/QTM350/blob/ea4f3125029d4befbfd8821c6937a681c4e041d1/photos/Screen%20Shot%202021-04-22%20at%203.30.43%20PM.png)

**Step Two: Using Amazon Translate**

After you have created your two buckets, folders, and stored all of your files, you are ready to begin translating the files. First, search for the Amazon Translate service and go to its portal. Click on "Batch translation", which you can find on the left-hand side. Once you are there, your screen should show you Translation jobs.

![picture](https://github.com/samho22/QTM350/blob/ea4f3125029d4befbfd8821c6937a681c4e041d1/photos/Screen%20Shot%202021-04-22%20at%203.30.49%20PM.png)

Click on the "Create job" button in the top right corner. This will lead you to a screen to create a translation job.

In the Job settings section, you will need to input a name for your transcription job and choose the source language and target language for your translations. Source language is the language your text(s) currently is and target language is the language you wish to translate the text(s) to. Our source language was English (en) and we wanted to translate our texts to Korean (ko).

![picture](https://github.com/samho22/QTM350/blob/ea4f3125029d4befbfd8821c6937a681c4e041d1/photos/Screen%20Shot%202021-04-22%20at%203.30.57%20PM.png)

In the "Input data" section, you will need to input the S3 location of the bucket and folder with the original phrases you would like translated. This is the first bucket and folder we created above. You can click "Select folder" to find the S3 location you are looking for. Then, you must indicate what format your files are.

![picture](https://github.com/samho22/QTM350/blob/ea4f3125029d4befbfd8821c6937a681c4e041d1/photos/Screen%20Shot%202021-04-22%20at%203.31.04%20PM.png)

In the Output data section, you will need to input the S3 location of the bucket and folder where you would like to store the new translations. This is the second bucket and folder(s) we have created. Again, you can click Select folder to find the S3 location you are looking for.

![picture](https://github.com/samho22/QTM350/blob/ea4f3125029d4befbfd8821c6937a681c4e041d1/photos/Screen%20Shot%202021-04-22%20at%203.31.12%20PM.png)

In our case, we do not use the Customization section but can learn more about this section on [AWS's walkthrough on batch translations](https://docs.aws.amazon.com/translate/latest/dg/how-custom-terminology.html). This section allows you to use custom terminology with your translation requests.

In the Access permissions section, you can either use an existing IAM role or create an IAM role. For our first translation job, we chose to create an IAM role giving them access to input and output S3 buckets, which we gave the role name "Translate". For the rest of our translation jobs, we used an existing IAM role which was the one that we had just created in our first translation job. For us, that role was called "AmazonTranslateServiceRole-Translate".

![picture](https://github.com/samho22/QTM350/blob/ea526253ff543a3a7bd331ee1740f1bea38a7e55/photos/Screen%20Shot%202021-04-22%20at%203.33.56%20PM.png)

After you have put in all of this information, click "Create job" at the bottom of the page. The translation job will now start working. Once it is complete, it will say completed in green in the status section.

![picture](https://github.com/samho22/QTM350/blob/ea526253ff543a3a7bd331ee1740f1bea38a7e55/photos/Screen%20Shot%202021-04-22%20at%203.34.21%20PM.png)

As you can see from the image above, we made eight different translation jobs. Each job translated the same phrases from our first bucket (in English) to our eight different languages we wanted to analyze: Spanish (Mexico), Portuguese, German, French, Hindi, Tagalog, Simplified Mandarin, and Korean. They were then stored in their respective folder in the second bucket we created. 

#### See the translations
Once the job is complete, you can find the translations in the bucket in which you indicated as the output location. You can do this by clicking on the job and then clicking the output file location in the bottom right corner, as in the image below. This will take you directly to the folder with all the new translated text files.


![picture](https://github.com/samho22/QTM350/blob/ea526253ff543a3a7bd331ee1740f1bea38a7e55/photos/Screen%20Shot%202021-04-22%20at%203.34.27%20PM.png)

![picture](https://github.com/samho22/QTM350/blob/ea526253ff543a3a7bd331ee1740f1bea38a7e55/photos/Screen%20Shot%202021-04-22%20at%203.34.34%20PM.png)

You can also go directly to the S3 bucket. Once you click on the folder you sent the data to, you will see a new folder and a temp file. Click on the folder file to see all of the new text files with the new translations. The folder you are clicking on is the same folder from the image above.

![picture](https://github.com/samho22/QTM350/blob/ea526253ff543a3a7bd331ee1740f1bea38a7e55/photos/Screen%20Shot%202021-04-22%20at%203.34.40%20PM.png)

By clicking on the text file, you can see all of it's properties. To see what is in this text file, click Object actions in the top right corner. Then you can open or download the text.

![picture](https://github.com/samho22/QTM350/blob/ea526253ff543a3a7bd331ee1740f1bea38a7e55/photos/Screen%20Shot%202021-04-22%20at%203.34.46%20PM.png)

![picture](https://github.com/samho22/QTM350/blob/ea526253ff543a3a7bd331ee1740f1bea38a7e55/photos/Screen%20Shot%202021-04-22%20at%203.34.52%20PM.png)

## Analysis of the Translations

We calculated accuracy of the AWS translations through word match using this comparison [website](https://countwordsfree.com/comparetexts). 

For more information, check out our [blog](https://qtm350finalprojectblog.s3.amazonaws.com/TricoloreProjectBlogFinal.html). 





## Architecture Overview 


![picture](https://github.com/samho22/QTM350/blob/ea526253ff543a3a7bd331ee1740f1bea38a7e55/Amazon%20Translate%20Architecture%20Diagram.jpg)

We first took our text of formal and slang phrases to be translated and uploaded the file into our project S3 bucket. Then, we used the Amazon Translate service to get a translation of all of the phrases in our 8 languages (Hindi, Tagalog, Mandarin, Korean, German, Portugese, French, Spanish). Next, the translated texts were put into a new S3 bucket. Then, we checked the accuracy of the translation by comparing them to a native speaker's translation. Lastly, created a data frame of the accuracy scores for each language and created data visualizations.

To view our full analysis, check out our [final analysis notebook ](https://github.com/samho22/QTM350/blob/main/QTM350TricoloreFinalProjectAnalysis.ipynb) in our repository.


## Authors 
Chelsey Kim

Vyas Muralidharan

Angela Guevarra

Samantha Ho
