# Silke's Weekly Meeting Notes

- [29 September 2026](#date-29-september-2026)
- [22 September 2026](#date-22-september-2026)
- [10 September 2026](#date-10-september-2026)
- [1 September 2026](#date-1-september-2026)
- [Template](#date-template)

## Date: 29 September 2026

#### What did you achieve?

- Made distance matrix of posteriers per scan.
    - Made t-sne plot using the distance matrix
- Made plots of the mean posteriers of positives, negatives, and plots based on the 10 best and worst performing postives and negatives
- Made plot of number of scans per patient
- Included variability in the max depth random forest plot
- Checked the performance of models trained only on data from one hospital
    - Hillerød performs worse than Herlev and Hvidovre
        - I have asked if there is something different about Hillerøds scanning setup
    - Dev set: 9 scans from Odense, 626 from Hvidovre, 190 from Herlev, 284 from Hillerød
    - Test set: 0 scans from Odense, 61 from Hvidovre, 23 from Herlev, 38 from Hillerød
- Tried using 200 slices evenly spread out across the middle 80% of the scan -> did not perform better

#### What did you struggle with?

-

#### What would you like to work on next week?

-

#### Where do you need help from Veronika?

- What should I do with the distance matrix?

#### Any other topics

-

#### Meeting agreements

- Replace with what you agreed on in the meeting

## Date: 22 September 2026

#### Who did you help this week?

- N/A

#### Who helped you this week?

- N/A

#### What did you achieve?

- Implemented the ultrasam feature extractor
- Tried random forrest and logistic regression
- Slicing in the sagittal view using the 200 middle slices
- Plotted the result for all volume slices
- Split the dataset into a development set and a test set
- Wrote some initial notes for methods section in overleaf
    - Currently working on a visualization for it
- Storing posteriers and brier score to metrics section

#### What did you struggle with?

- Plotted the results of random forrest with different max depths, but it did not look as expected + random forrest takes a long time to run
- How to aggregate the results of each slice into one result for the volume. Now I just use the mean of all the slices, but this might not be the best way?

#### What would you like to work on next week?

- Find better hyperparameters for the classification models
- Feature extraction -> Slicing
- Write more on the methods section and finishing the visualization
- Implement the model based on meta data

#### Where do you need help from Veronika?

- Show plots for results per slices -> whats the best way to aggregate?
- How to slice when the volumes have different dimensions

#### Any other topics

- N/A

#### Meeting agreements

- Make a distance matrix of the postiers from the scans
    - Visualize using t-ssne
- Make a plot of number of scans per patient
- Include variability in the random forest per depth plot (error bar)

## Date: 10 September 2026

#### Who did you help this week?

- N/A

#### Who helped you this week?

- Luisa helped with various questions related to working on gefion

#### What did you achieve?

- Read the USF-MAE and UltraSam papers
- Added the functionality to extract features using the USF-MAE model
    - Tried to run the full pipeline using these features which gave higher AUC scores for 3/8 musa features, 1/8 was the same and 4/8 was better with DINOv3
- Found out that we were slicing in the transversal view, which is most likely not optimal
    - Plotted some volumes to try to find out which axis corresponds to the sagital view
    - Extracted features using the sagittal view which is giving better results for some musa features

#### What did you struggle with?

- Gefion shutting down while working
- Deciding how to make the create the models from the papers (e.g. using timm, mmpretrain or facebook-mae)
    - I think I will end up using timm for all 3, and load the weights from the paper into the model

#### What would you like to work on next week?

- Add functionality to extract features using the UltraSam model, and test it
- Experiment more with slicing
    - Now we use all slices and use mean pooling
    - When you look at the slices it seems like the middle 3rd is the most relevant, maybe just use this?
    - Maybe use max pool instead of average?
    - Maybe use different slicing views at the same time?
- Try plotting the extracted features using dimensionality reduction
- Try random forrest

#### Where do you need help from Veronika?

- How directly can I use my research paper in my thesis?
- Any suggestions related to slicing?
- Loss in multi label classifier

#### Any other topics

- The 23rd works. Do you have time between 13 and 15?

#### Meeting agreements

- Make an external test set that will only be used for testing at the end of the thesis
    - Add this to the methodology
- Make a visualization of the setup (train test split, feature extractor models used, classification model ect) that can be added to the thesis
- Make a github/gitlab repo that will be accesible by Veronika and censor, that is updated regularly
- Do not use their excisting code, make everything from scratch
- Do not use the average feature vector of all slices. Have one feature vector per slice that will be input to the classification model



## Date: 1 September 2026

#### Who did you help this week?

- N/A

#### Who helped you this week?

- Luisa from COPL

#### What did you achieve?

- Watched youtube videos about transformers and vision transformers
- Read the paper attention is all you need
- First day at Hvidovre
    - I got my computer and I have access to everything
    - Update on their status:
        - Multi label -> A binary model per MUSA feature
        - Pretrained vision transformer feature extractor (not finetuned on their data at the moment)
        - Logistic regression classifier
        - They are currently using this vision transformer as the feature extractor: https://arxiv.org/abs/2508.10104
        - They want me to try these two and see how they compare
            - https://link.springer.com/article/10.1007/s11548-025-03517-8
            (not to be confused with [this](https://www.sciencedirect.com/science/article/abs/pii/S0957417425038382) paper that has a similar name )
            - https://www.sciencedirect.com/science/article/pii/S1746809426008670?via%3Dihub
        - Accuracy is currently around 0.6 for most of the MUSA features but one perform better, around 0.8

#### What did you struggle with?

- I find transformers hard to fully grasp but I have gotten a much better understanding of them this week

#### What would you like to work on next week?

- Read the 3 papers
- Get an overview of the code base

#### Where do you need help from Veronika?

- Do you think a comparison of the 3 models is a good starting point?

#### Any other topics

- Overall timeline. Is it realistic to hand in before christmas?
- When will we know exam date?


## Date: Template

#### Who did you help this week?

- Replace this text with a one/two sentence description of who you helped this week and how.

#### Who helped you this week?

- Replace this text with a one/two sentence description of who helped you this week and how.

#### What did you achieve?

- Replace this text with a bullet point list of what you achieved this week.
- It's ok if your list is only one bullet point long!

#### What did you struggle with?

- Replace this text with a bullet point list of where you struggled this week.
- It's ok if your list is only one bullet point long!

#### What would you like to work on next week?

- Replace this text with a bullet point list of what you would like to work on next week.
- It's ok if your list is only one bullet point long!
- Try to estimate how long each task will take.

#### Where do you need help from Veronika?

- Replace this text with a bullet point list of what you need help from Veronika on.
- It's ok if your list is only one bullet point long!
- Try to estimate how long each task will take.

#### Any other topics

This space is yours to add to as needed.

## Date: Template

#### What did you achieve?

- Replace this text with a bullet point list of what you achieved this week.
- It's ok if your list is only one bullet point long!

#### What did you struggle with?

- Replace this text with a bullet point list of where you struggled this week.
- It's ok if your list is only one bullet point long!

#### What would you like to work on next week?

- Replace this text with a bullet point list of what you would like to work on next week.
- It's ok if your list is only one bullet point long!
- Try to estimate how long each task will take.

#### Where do you need help from Veronika?

- Replace this text with a bullet point list of what you need help from Veronika on.
- It's ok if your list is only one bullet point long!
- Try to estimate how long each task will take.

#### Any other topics

This space is yours to add to as needed.

#### Meeting agreements

- Replace with what you agreed on in the meeting