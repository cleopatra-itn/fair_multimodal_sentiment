# A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment Analysis Methods

The repository for the splits and code used in the paper
> Gullal S. Cheema, Sherzod Hakimov, Eric Müller-Budack and Ralph Ewerth “A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment Analysis Methods“, 
*Proceedings of the 2021 Workshop on Multi-Modal Pre-Training for Multimedia Understanding (MMPT ’21), August 21, 2021,Taipei, Taiwan.

## Splits
- 10 Fold Train/Val/Test splits provided in data/ for MVSA-single and MVSA-multiple.
- valid_pairlist.txt format is `file_id (filename), multimodal label, text label, image label`
- 0 (Neutral), 1 (Positive), 2 (Negative)
- Split file rows point to the line number in valid_pairlist.txt (0-indexed)
- `multimodal label` is used for training and evaluating all the models.



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
