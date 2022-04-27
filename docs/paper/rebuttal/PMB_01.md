# PMB-113146: Responses to referee comments

## General comments
<span style="color:dimgray">*We would like to thank all reviewers for the encouraging comments that helped us to improve the manuscript.*</span>

### Highlighting of changes
<span style="color:dimgray">*Text that was changed (added, removed or replaced) is highlighted in blue.*</span>

### Reconstruction of real TOF PET data
<span style="color:dimgray">*As requested by both reviewers, we have added the results of the reconstruction of a real data set (NEMA image quality phantom) acquired on GE DMI 4 ring TOF PET/CT. The results are shown in Figure 7. Note that for this data set we used a LM-PDHG reconstruction with 20000 iterations and 1 subset as reference. Doing the same with the sinogram PDHG was computationally not feasible.*</span>

### Structured abstract
<span style="color:dimgray">*The abstract is now structured into Objective, Approach, Main results, Significance*</span>


* * *

## Point-by-point reply Referee 1

**Summary: The paper "Fast and memory-efficient reconstruction of sparse Poisson data in listmode with non-smooth priors with application to time-of-flight PET" by Schramm and Holler discusses an algorithm for PET reconstruction which can handle modern reconstruction models and is adapted to the data structure (mostly sparsity) of modern time-of-flight PET data. They formulate the problem as a maximum-a-posteriori estimate which fits the template of the SPDHG algorithm. Moreover, they notice that certain variables do not have to be computed by the algorithm and can be computed by hand, thereby accelerating the algorithm. The algorithm is well presented making it easy for the interested reader to code it themselves. Extensive numerical examples show that the algorithm is as fast as SPDHG on binned data using only a fraction of the memory. The paper is well written and the topic is timely and interesting to the image reconstruction community. The discussion and conclusions are appropriate. Overall, a good paper that could be accepted as is. Nevertheless, I have a couple of recommendations that would make the paper even stronger**

<span style="color:dimgray">*We would like to thank the reviewer for the encouraging summary of our work.*</span>

## Main
1. On page 5 they argue that they can write down the listmode likelihood and therefore can apply SPDHG (with all its guarantees) out of the box. It would be good to write this down explicitly in the paper and not just mentioning it. This is also necessary for the reader to understand why their stepsize conditions are sufficient for convergence.
<span style="color:dimgray">*We added the listmode re-formulation of the sinogram-based optimization problem in appendix A and refer to it on page 5.*</span>

2. Convergence speed of an algorithm is likely affected by the exact data and image content. It would strengthen the paper a lot to have one clinical data set to confirm the same conclusions on real data. 
<span style="color:dimgray">*We added a reconstruction of a NEMA image phantom acquired on a GE Discovery MI 4 ring TOF PET system. Please see the general comment at the beginning*</span>

3. Figure 5: Why only 224 subsets? Intuitively, there is no upper bound on the subsets to be used within this setting. Why not take 1000s or even more? At some point one may see some diminishing returns as can be observed for TV here but DTV looks like can be made a lot faster.
<span style="color:dimgray">*The reason why we applied "only" 224 is that the simulated scanner has 224 views in the sinogram. Since traditionally subsets in sinogram-based reconstructions are based on views and since we always wanted to compare the LM version against sinogram versions, we did not use more than 224 subsets (1 view per subset with the simulated scanner). For the LM approach there is indeed no upper bound on the number of subsets to use and the "optimal" number of subsets will probably depend on the total number of counts that were acquired (e.g. choosing the number of susbets such that a fixed number of counts is in one subset seems a reasonable idea) Since our manuscript is already quite long, we would like to keep the investigation of the ideal of subsets vs the acquired number of counts and the object for future research. We have added sentence in the discussion.*</span>

## Minor (mainly typos)

1. P2 l12: The convex functions in this argument are D and beta * R but their notation suggests these were x \to D(Px + s) and x \to beta * R(Kx)
<span style="color:dimgray">*We have updated the notation accordingly.*</span>

2. P2 l41: "10 complete projections … are sufficient". This sounds to be rather optimistic and perhaps wrong. Better "10 complete projections … can be sufficient"
<span style="color:dimgray">*"are" was replaced by "can be"*</span>

3. P2 l45 positive => positive-definite
<span style="color:dimgray">*positive-definite is now used*</span>

4. Eq 5: i missing: it should be prox_{D_i\*}^{S_i}
<span style="color:dimgray">*missing "i"s in "S_i"s were added*</span>

5. P7 l 39: monitored => monitored
<span style="color:dimgray">*We were not sure what the typo was (seems to be two times the same word in the issue above).*</span>

***

## Point-by-point reply Referee 2

**Summary: In this manuscript, the authors explored the list-mode extension of the SPDHG algorithm for PET image reconstruction with TOF information. To solve the problem that TOF PET reconstruction from sinogram data is highly memory required and long computation time needed, direct projection from list-mode data was proposed to replace the sinogram projection in the SPDHG algorithm. The memory requirements of the proposed LM-SPDHG were significantly reduced. This reviewer only has a few comments for seeking clarifications:**

1. Some symbols in some equations that were not clearly expressed such as P , x   in Equation(1)  may need to be in bold？
<span style="color:dimgray">*In our notation, operators are represented by capital latin charactres (e.g. P, K), vectors are represented by small latin characters (e.g. x, y, s) and scalars are represented by small greek letters. This convention is often used in mathematics as e.g. in the original PDHG paper by Chambolle and Pock.*</span>

2. A little more explanation on line4 and line5 in Algorithm2 is needed. Does “event count” mean the counts of the same LOR?
<span style="color:dimgray">*The "event counts" means counts in the same measurement bin. For TOF PET that means counts on the same geometrical LOR in the same TOF bin. We have added a footnote to explain this more clearly. Line 5 is explained in detail after Equation (9). and line 4 corresponds to the initialization of y in Eq. (7) using the LM forward projector.*</span>

3. In Fig.2, although the cost and PSNR show that the warm start performs better, it's hard to tell which is better in pictures, maybe the error image is preferable?
<span style="color:dimgray">*The difference images were added to Figure 2. It is correct that visually the differences between the cold and ward started sinogram SPDHGs is "small". However, as stated by the reviewer, from the PSNR plot it can be seen that the warm-started SPDHG is "closer" to the reference reconstruction at all iterations. Note, the main advantage of the proposed warm start for x and y is not that is leads to much faster convergence. Instead, the initialization of y according to Eq. (7) (where all bins without data, get initialized with the optimal value 1), allows the neglect those bins during the iterations as explained before Eq. (6). This in the end means that during the iterations of LM-SPDHG the forward and adjoint operator do not have to evaluated for data bins where no counts where measured.*</span>

4. In Fig.4, the subsets for list-mode EM-TV are 1, 7, 28. Is 7 the most appropriate subset number of list-mode EM-TV? It would be helpful to provide an explanation on the choice of subset number of the comparison method.
<span style="color:dimgray">*In our experience, the most appropriate number of subsets in EM-TV heavily depends on the number of counts in the measurement. In general, for high count scans more subsets can be used compared to low-count scan. Not that as with OSEM, EM-TV with subsets (OSEM-TV) does not convergence to he optimal solution but instead stays on a limit cycle around the optimal solution as seen in the PSNR plots in Fig. 5a and 5b for 28 subsets (purple curve). The distance of this limit cycle from the optimal point depends on the number of subsets used and the number of acquired counts. Since OSEM-TV with more than 1 subset does not converge to the optimal solution and since the convergence EM-TV without subsets is usually slow, we just added the cases of 7 and 28 subsets as a reference. For the experiments shown in Fig 5a and 5b that were stopped at 100 iterations, the optimal subset number should be somewhere between 7 and 28. As mentioned about in the 3rd comment we would like to leave the discussion on the "optimal number of listmode subsets" for future research.*</span>

5. Since the paper focus on reducing the memory requirements of optimization-based TOF-PET reconstruction algorithms. It may need to mention the memory requirements of list-mode EM-TV.
<span style="color:dimgray">*The memory requirements for listmode EMTV were added to Table 2.*</span>

6. The reconstruction times of the proposed method and comparison methods may need to be quantified.
<span style="color:dimgray">*As written in the discussion, the total reconstruction times heavily depend on the specific implementation and computing hardware used. Since in this proof-of-concept work, we still used a hybrid computing approach where only the PET forward and backprojections were computed on the GPU and thus memory transfer between host and GPU is one major bottleneck (this is because the GPU projectors of the "parallelproj" library we used here were setup for the hybrid computing approach). This is why we would prefer not to compare actual reconstructions times between LM-SPDHG, SPDHG and EM-TV. Note that with data from state-of-the art TOF PET scanners and state-of-the-art GPUs (with GPU memory between 12-24 GB), all algorithms using LM data can be implemented purely on GPUs whereas this is not possible for sinogram-based algorithms due to their memory requirements as discussed in table 1. Based on the fact that the overall speed of convergence between LM-SPDHG and SPDHG is very similar, we know that LM-SPDHG is faster compared to SPDHG in cases where the computation time of the listmode projections are faster compared to sinogram projections when using similar implementations and computing hardware. However, the fact that the LM-based algorithms can be implemented purely on GPUs will dramatically decrease the actual reconstruction time.*</span>

7. It would be helpful to provide a description of the experimental environment.
<span style="color:dimgray">*The experimental environment (used parallelproj projectors and V100 GPU) are now described before subsection "SPDHG using a warm start vs a cold start"* in section 3. Upon potential publication of this work, we will also provide an open-source reference implementation of LM-SPDHG in python using the parallelproj projectors.</span>

8. The major limitation of this paper is that it lacks of clinical data validations. It would be better if there were clinical data validations.
<span style="color:dimgray">*We added a reconstruction of a NEMA image phantom acquired on a GE Discovery MI 4 ring TOF PET system. Please see the general comment at the beginning.*</span>
