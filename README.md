# A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment Analysis Methods

The repository for the splits and code used in the paper
> Gullal S. Cheema, Sherzod Hakimov, Eric Müller-Budack and Ralph Ewerth “A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment Analysis Methods“, 
*Proceedings of the 2021 Workshop on Multi-Modal Pre-Training for Multimedia Understanding (MMPT ’21), August 21, 2021,Taipei, Taiwan.

Paper is available on arXiv: https://arxiv.org/abs/2106.08829

## Data and Environment
- Available at https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
- Download and extract images and texts into separate folders in `data/mvsa_single/` and `data/mvsa_multiple/`
- Create an environment using environment.yml with conda
## Splits
- 10 Fold Train/Val/Test splits provided in data/ for MVSA-single and MVSA-multiple.
- valid_pairlist.txt format is `file_id (filename), multimodal label, text label, image label`
- 0 (Neutral), 1 (Positive), 2 (Negative)
- Split file rows point to the line number in valid_pairlist.txt (0-indexed)
- `multimodal label` is used for training and evaluating all the models.

## Extract Features
- Download pretrained models to `pre_trained` : [places](https://drive.google.com/file/d/1ARP8GS5LMGYc8T8lFTuYkBl9I9kJoIiL/view?usp=sharing), [emotion](https://drive.google.com/file/d/1sWx3ze8XfZEGf-kPcmiYpY9EOzugdzgu/view?usp=sharing).
- Download face expression features into `features` folder from [mvsa_single](https://drive.google.com/file/d/1akwt0-8fqea2WinTDWAk68qRDkYqEEiR/view?usp=sharing), [mvsa_multiple](https://drive.google.com/file/d/1h6A8KjhVEy2QSEBGQS8kH2HQxEa__a2S/view?usp=sharing) 
- Extract image features: `python feature_extraction/extract_img_feats.py --vtype imagenet --mvsa single --ht False`
- Extract text features: `python feature_extraction/extract_txt_feats.py --btype robertabase --mvsa single --ht True`

## Train and evaluate models
- With one type of visual feature and BERT feature: `python train/multi_mlp_2mod.py --vtype clip --ttype clip --mvsa single --ht True --bs 32 --split 1`


## Cite
If you find the code and paper useful, kindly give a star and cite us as below:

```
@inproceedings{DBLP:conf/mir/CheemaHME21,
  author    = {Gullal S. Cheema and
               Sherzod Hakimov and
               Eric M{\"{u}}ller{-}Budack and
               Ralph Ewerth},
  editor    = {Bei Liu and
               Jianlong Fu and
               Shizhe Chen and
               Qin Jin and
               Alexander G. Hauptmann and
               Yong Rui},
  title     = {A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment
               Analysis Methods},
  booktitle = {MMPT@ICMR2021: Proceedings of the 2021 Workshop on Multi-Modal Pre-Training
               for Multimedia Understanding, Taipei, Taiwan, August 21, 2021},
  pages     = {37--45},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3463945.3469058},
  doi       = {10.1145/3463945.3469058},
  timestamp = {Wed, 06 Oct 2021 14:51:08 +0200},
  biburl    = {https://dblp.org/rec/conf/mir/CheemaHME21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
