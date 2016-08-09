Relation Knowledge Transfer for Zero-Shot Learning
=============
This is the implement of  paper "Relation Knowledge Transfer for Zero-Shot Learning "(Donghui Yanan+ 2015 AAAI)
[http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11802]

Get The Support
-------------------------
Download the SLEP4.1 on http://www.yelab.net/software/SLEP/ and replace the dir SLEP_package_4.1 in support 

How to Run
-------------------------
1. Download the Dataset on google drive (https://drive.google.com/open?id=0B2u0zk7WYuVpNDFZSUxYNzlqRWs), and put them into the dataset dir

2. Open the Matlab, make the current dir to be the Relational-Knowledge-Transfer-for-ZSL, run "code/ZSL/v-release/DAL_F2K2C_main.m"

Citation
-------------------------
If you find the dataset and toolbox useful in your research, please consider citing:
<pre>
@paper{AAAI1611802,
    author = {Donghui Wang and Yanan Li and Yuetan Lin and Yueting Zhuang},
	title = {Relational Knowledge Transfer for Zero-Shot Learning},
	conference = {AAAI Conference on Artificial Intelligence},
	year = {2016},
	keywords = {},
	abstract = {General zero-shot learning (ZSL) approaches exploit transfer learning via semantic knowledge space. In this paper, we reveal a novel relational knowledge transfer (RKT) mechanism for ZSL, which is simple, generic and effective. RKT resolves the inherent semantic shift problem existing in ZSL through restoring the missing manifold structure of unseen categories via optimizing semantic mapping. It extracts the relational knowledge from data manifold structure in semantic knowledge space based on sparse coding theory. The extracted knowledge is then transferred backwards to generate virtual data for unseen categories in the feature space. On the one hand, the generalizing ability of the semantic mapping function can be enhanced with the added data. On the other hand, the mapping function for unseen categories can be learned directly from only these generated data, achieving inspiring performance. Incorporated with RKT, even simple baseline methods can achieve good results. Extensive experiments on three challenging datasets show prominent performance obtained by RKT, and we obtain 82.43% accuracy on the Animals with Attributes dataset.},

	url = {http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11802}
}
</pre>

Any Question?
-------------------------
Send Email to Me

ynli@zju.edu.cn