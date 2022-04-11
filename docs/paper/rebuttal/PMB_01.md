NIMG-20-308

# PMB-113146: Responses to referee comments

## General comments
<span style="color:dimgray">*We would like to thank all reviewers for the encouraging comments that helped us to improve the manuscript.*</span>

### Clinical data
<span style="color:dimgray">*Foo bar*</span>


* * *

## Point-by-point reply Referee 1

**Summary: The paper "Fast and memory-efficient reconstruction of sparse Poisson data in listmode with non-smooth priors with application to time-of-flight PET" by Schramm and Holler discusses an algorithm for PET reconstruction which can handle modern reconstruction models and is adapted to the data structure (mostly sparsity) of modern time-of-flight PET data. They formulate the problem as a maximum-a-posteriori estimate which fits the template of the SPDHG algorithm. Moreover, they notice that certain variables do not have to be computed by the algorithm and can be computed by hand, thereby accelerating the algorithm. The algorithm is well presented making it easy for the interested reader to code it themselves. Extensive numerical examples show that the algorithm is as fast as SPDHG on binned data using only a fraction of the memory. The paper is well written and the topic is timely and interesting to the image reconstruction community. The discussion and conclusions are appropriate. Overall, a good paper that could be accepted as is. Nevertheless, I have a couple of recommendations that would make the paper even stronger**

<span style="color:dimgray">*We would like to thank the reviewer for the encouraging summary of our work.*</span>

## Main
1. On page 5 they argue that they can write down the listmode likelihood and therefore can apply SPDHG (with all its guarantees) out of the box. It would be good to write this down explicitly in the paper and not just mentioning it. This is also necessary for the reader to understand why their stepsize conditions are sufficient for convergence.
<span style="color:dimgray">*Our reply*</span>

2. Convergence speed of an algorithm is likely affected by the exact data and image content. It would strengthen the paper a lot to have one clinical data set to confirm the same conclusions on real data. 
<span style="color:dimgray">*Our reply*</span>

3. Figure 5: Why only 224 subsets? Intuitively, there is no upper bound on the subsets to be used within this setting. Why not take 1000s or even more? At some point one may see some diminishing returns as can be observed for TV here but DTV looks like can be made a lot faster.
<span style="color:dimgray">*Our reply*</span>

## Minor (mainly typos)

1. P2 l12: The convex functions in this argument are D and beta * R but their notation suggests these were x \to D(Px + s) and x \to beta * R(Kx)
<span style="color:dimgray">*Our reply*</span>

2. P2 l41: "10 complete projections … are sufficient". This sounds to be rather optimistic and perhaps wrong. Better "10 complete projections … can be sufficient"
<span style="color:dimgray">*Our reply*</span>

3. P2 l45 positive => positive-definite
<span style="color:dimgray">*Our reply*</span>

4. Eq 5: i missing: it should be prox_{D_i*}^{S_i}
<span style="color:dimgray">*Our reply*</span>

5. P7 l 39: monitored => monitored
<span style="color:dimgray">*Our reply*</span>

***

## Point-by-point reply Referee 2

**Summary: In this manuscript, the authors explored the list-mode extension of the SPDHG algorithm for PET image reconstruction with TOF information. To solve the problem that TOF PET reconstruction from sinogram data is highly memory required and long computation time needed, direct projection from list-mode data was proposed to replace the sinogram projection in the SPDHG algorithm. The memory requirements of the proposed LM-SPDHG were significantly reduced. This reviewer only has a few comments for seeking clarifications:**

1. Some symbols in some equations that were not clearly expressed such as P , x   in Equation(1)  may need to be in bold？
<span style="color:dimgray">*Our reply*</span>

2. A little more explanation on line4 and line5 in Algorithm2 is needed. Does “event count” mean the counts of the same LOR?
<span style="color:dimgray">*Our reply*</span>

3. In Fig.2, although the cost and PSNR show that the warm start performs better, it's hard to tell which is better in pictures, maybe the error image is preferable?
<span style="color:dimgray">*Our reply*</span>

4. In Fig.4, the subsets for list-mode EM-TV are 1, 7, 28. Is 7 the most appropriate subset number of list-mode EM-TV? It would be helpful to provide an explanation on the choice of subset number of the comparison method.
<span style="color:dimgray">*Our reply*</span>

5. Since the paper focus on reducing the memory requirements of optimization-based TOF-PET reconstruction algorithms. It may need to mention the memory requirements of list-mode EM-TV.
<span style="color:dimgray">*Our reply*</span>

6. The reconstruction times of the proposed method and comparison methods may need to be quantified.
<span style="color:dimgray">*Our reply*</span>

7. It would be helpful to provide a description of the experimental environment.
<span style="color:dimgray">*Our reply*</span>

8. The major limitation of this paper is that it lacks of clinical data validations. It would be better if there were clinical data validations.
<span style="color:dimgray">*Our reply*</span>
