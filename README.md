# A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment Analysis Methods

The repository for the splits and code used in the paper
> Gullal S. Cheema, Sherzod Hakimov, Eric Müller-Budack and Ralph Ewerth “A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment Analysis Methods“, 
*Proceedings of the 2021 Workshop on Multi-Modal Pre-Training for Multimedia Understanding (MMPT ’21), August 21, 2021,Taipei, Taiwan.

## Splits
- 10 Fold Train/Val/Test splits provided in data/ for MVSA-single and MVSA-multiple.
- valid_pairlist.txt format is `file_id (filename), multimodal label, text label, image label`
- 0 (Neutral), 1 (Positive), 2 (Negative)
- Split file rows point to the line number in valid_pairlist.txt
- `multimodal label` is used for training and evaluating all the models.



```
@article{DBLP:journals/corr/abs-2106-08829,
  author    = {Gullal S. Cheema and
               Sherzod Hakimov and
               Eric M{\"{u}}ller{-}Budack and
               Ralph Ewerth},
  title     = {A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment
               Analysis Methods},
  journal   = {CoRR},
  volume    = {abs/2106.08829},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.08829},
  archivePrefix = {arXiv},
  eprint    = {2106.08829},
  timestamp = {Tue, 29 Jun 2021 16:55:04 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-08829.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
