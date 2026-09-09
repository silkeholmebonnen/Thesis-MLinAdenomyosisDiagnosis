# Silke's Weekly Meeting Notes

- [10 September 2026](#date-10-september-2026)
- [1 September 2026](#date-1-september-2026)
- [Template](#date-template)

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

#### Where do you need help from Veronika?

- How directly can I use my research paper in my thesis?
- Any suggestions related to slicing?

#### Any other topics

- The 23rd works. Do you have time between 13 and 15?

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