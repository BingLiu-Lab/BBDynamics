
# BBDynamics

This repository hosts the core code for the manuscript ***Deciphering Complex Brain Spatiotemporal Dynamics Shaping Diverse Human Behavior***.

*Please note that this repository is a preliminary version. The official version will be released once it has been thoroughly peer-reviewed*.


## Overview

Our study is based on the 3T-rfMRI data and diverse behavior phenotyping. We aimed to propose a comprehensive framework that delineates the complex brain spatiotemporal dynamics in relation to diverse human behaviors. For detailed procedures and results, please refer to the manuscript. 

We here provide the core code and foundational introduction for following results:

* [Topographic architecture of whole-brain spontaneous dynamics](./UMAP_analysis/): UMAP embedding based on whole-brain dense time-series features matrix of brain spontaneous dynamics.

* [Key 44 rfMRI BOLD time-series features](./44_timeseries_features) : ICC computation, computation methods and references for the 44 time-series features.

* [CCA analysis](./CCA_Analysis) : Brain-behavior mode, specific spatiotemporal dynamics patterns linked to human behavior.



## Software and Code Utilized
The software and code used in our study are publicly accessible. Detailed information, including specific versions, is available in ***Methods*** section. For guidelines on usage or the associated LICENSE, please consult and adhere to the specifications in the source package or the software documentation. We sincerely thank the developers of these open-source software or packages.


```
hctsa_1.06
umap-learn_0.5.0
scikit-learn_0.23.2
pingouin_0.5.2
statsmodels_0.12.0
SOLAR_9.0.0
```


Specifically, the [hctsa](https://github.com/benfulcher/hctsa/) toolbox, which we leveraged to extract rich, interdisciplinary time-series features from whole-brain spontaneous dynamics. This toolbox has been consistently well-maintained, establishing itself as a mature and prevalently employed instrument for temporal dynamic analysis. For those seeking in-depth instructions and explanations regarding this toolbox, the developers offer extensive tutorials and documentation：

* Research article ：B.D. Fulcher and N.S. Jones. hctsa: A computational framework for automated time-series phenotyping using massive feature extraction. Cell Systems 5, 527 (2017).

* Official GitHub project ：https://github.com/benfulcher/hctsa

* Comprehensive tutorial on using the hctsa toolbox：https://hctsa-users.gitbook.io/hctsa-manual/


