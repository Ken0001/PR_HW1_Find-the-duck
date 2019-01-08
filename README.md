# PR_HW1_Find-the-duck
本作業是用Python寫的
<br>1.首先會先從農場原圖(full_duck.jpg)中提取出鴨子的圖片和非鴨子的圖片，再用Python的OpenCV套件把這些提取出來的圖片讀進去並用list存起來。
>     cv2.imread

2.接下來要建立Likelihood model所以要算鴨子像素和非鴨子像素的mean和covariance matrix，
>     np.mean & np.cov

3.根據Bayes Decision Rule去計算鴨子像素和非鴨子像素中的p(x|duck)和p(x|nonduck)，用實驗的方式去得出最好的θ
```python
     # Compute the p(x|Duck) for all trained duck & nonduck pixels
     duck_given_duck = mvn.pdf(duck_pixels, duck_mean, duck_cov)
     non_duck_given_duck = mvn.pdf(non_duck_pixels, duck_mean, duck_cov)
     
     # Compute the p(x|nonDuck) for all trained duck & nonduck pixels
     duck_given_non_duck = mvn.pdf(duck_pixels, non_duck_mean, non_duck_cov)
     non_duck_given_non_duck = mvn.pdf(non_duck_pixels, non_duck_mean, non_duck_cov)
     
     # Compute the likelihood ratio p(x|Duck)/p(x|Non-Duck) for all duck & nonduck pixels
     likelihood_ratio1 = duck_given_duck/duck_given_non_duck
     likelihood_ratio2 = non_duck_given_duck/non_duck_given_non_duck
```
4.求出θ後再把原圖讀進來計算是鴨子的像素的機率有沒有大於這個最好的θ，再把這些像素畫出來其他的地方則黑掉，就可以得到output了<p>
Input & Output: <br>
<img src="full_duck.jpg" width=250>           <img src="duck_only.jpg" width=250> 
<!--![Input](/full_duck.jpg)  ![Output](/duck_only.jpg)-->
