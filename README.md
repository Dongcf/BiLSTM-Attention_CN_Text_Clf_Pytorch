# Data Scientist Evaluation Test
## There are totally 8 questions.  Try your best to work out the answer for each question. If you have no previous education background for a particular question, you may skip it.

>1.根据大数定理可知，当n=1000次抽样，可以认为是服从正太分布的
$$
\begin{matrix}
\mu_A = & mean(A) = 23 \\
\sigma_A = & SD(A) = 5 
\end{matrix} 
$$  则 ：$$ 
Z = \frac{x -\mu}{\sigma} = \frac{10 - 23}{5} = -2.6 
$$
 
>2.1
根据特征向量的定义有：
$$
\left |\begin{array}{cccc}
A - \lambda E
\end{array}\right| = 0 \tag{2.1.1}
$$ $$
\left |\begin{array}{cccc}
cosh(\alpha) - \lambda & \rho sinh(\alpha) \\
\frac{1}{\rho} sinh(\alpha) & cosh(\alpha) - \lambda
\end{array}\right| = 0  \tag{2.1.2}
$$ $$
(cosh(\alpha) - \lambda)^2 - sinh^2(\alpha) = 0   \tag{2.1.3}
$$ $$
(cosh(\alpha) - \lambda + sinh(\alpha))(cosh(\alpha) - \lambda - sinh(\alpha)) = 0  \tag{2.1.4}
$$ 得到特征值： $$
\lambda_1 = sinh(\alpha) +cosh(\alpha)  
$$ $$
\lambda_2 = cosh(\alpha) - sinh(\alpha) 
$$ 求特征向量 $$
(A - \lambda E)\vec{x} = 0 \tag{2.1.5}
$$ $$
(A - \lambda_1 E) \sim 
\left (\begin{array}{cccc}
-sinh(\alpha ) & \rho sinh(\alpha) \\
\frac{1}{\rho}sinh(\alpha) & -sinh(\alpha)
\end{array}\right) \sim \left (\begin{array}{cccc}
0 & 0 \\
\frac{1}{\rho}sinh(\alpha) & -sinh(\alpha)
\end{array}\right) \tag{2.1.6}
$$ 得 $$
\left (\begin{array}{cccc}
0 & 0 \\
\frac{1}{\rho}sinh(\alpha) & -sinh(\alpha)
\end{array}\right) \vec{x} = 0 
$$ 解得 $$
P_1 = \left (\begin{array}{cccc}
\rho \\
1
\end{array}\right)$$
$$ \therefore kP_1 是 \lambda_1 的特征向量(k \neq 0)$$
同理，将 $ \lambda_2 = cosh(\alpha) - sinh(\alpha) $ 带入（2.1.5）解得
$ P_2 = \left (\begin{array}{cccc}
\rho \\
-1
\end{array}\right) $
$$ \therefore kP_2 是 \lambda_2 的特征向量(k \neq 0)$$


>2.2
根据特征向量的定义有：
$$
\left |\begin{array}{cccc}
B - \lambda E
\end{array}\right| = 0 \tag{2.2.1}
$$ $$
\left |\begin{array}{cccc}
cos(\alpha) - \lambda & -\rho sin(\alpha) \\
\frac{1}{\rho} sin(\alpha) & cos(\alpha) - \lambda
\end{array}\right| = 0  \tag{2.2.2}
$$ $$
(cos(\alpha) - \lambda)^2 - sin^2(\alpha) = 0   \tag{2.2.3}
$$ $$
(cos(\alpha) - \lambda)^2 = -sin(\alpha)  \tag{2.2.4}
$$ 得到特征值： $$
\lambda_1 =\lambda_2 = cos(\alpha) 
$$ 求特征向量 $$
(B - \lambda E)\vec{x} = 0 \tag{2.2.5}
$$ $$
(B - \lambda E) = 
\left (\begin{array}{cccc}
0 & -\rho sin(\alpha) \\
\frac{1}{\rho}sin(\alpha) & 0
\end{array}\right)\tag{2.2.6}
$$ 得 $$
\left (\begin{array}{cccc}
0 & -\rho sin(\alpha) \\
\frac{1}{\rho}sin(\alpha) & 0
\end{array}\right) \vec{x} = 0 
$$ 解得 $$
P = \left (\begin{array}{cccc}
\pi \\
\pi
\end{array}\right)$$
$$ \therefore kP = k \left (\begin{array}{cccc}
\pi \\
\pi
\end{array}\right) 是矩阵B的特征向量(k \neq 0)$$

>3
$$
\begin{matrix}
\int_1^{\sqrt{e}} {\frac{arcsin(lnx)}{x}} \,{\rm d}x = \left . (lnxarcsin(lnx) + \sqrt{1 - (lnx)^2} \right|_1^{\sqrt{e}} \\
=(ln \sqrt{e}arcsin(ln\sqrt{e}) + \sqrt{1 - (ln\sqrt{e})^2} ) - ln1*arcsin(ln1) - \sqrt{(1 - (ln1)^2)} \\
= \frac{1}{2}arcsin\frac{1}{2} + \sqrt{1 - \frac{1}{4}} - 0 - 1 = \frac{\pi}{12} + \frac{\sqrt{3} - 2}{2}
\end{matrix}
$$

>4 
令A表示件“检测反应为阳性”
令C表示事件“被检测患者有疾病”
则 
$$
\begin{matrix}
P(A|C) = 0.85 \\
P(A|\overline{C}) = 0.2 \\
P(C) = 0.02 \\
P(C|A) = \frac{P(A|C)P(C)}{P(A|C)P(C) + P(A|\overline{C})P(\overline{C})} \\
= \frac{0.85 \times 0.02 }{0.85 \times 0.02 + 0.2 \times 0.98} = 0.0798
\end{matrix}$$
该被检测者确实是患有疾病的概率是0.0798

>5
(a) Consider first the case a>0.Using the definition of the Fourier transform and a change of variables
$$ 
\begin{matrix}
F(f(ax))(\omega) = \frac{1}{2\pi}\int_{-\infty}^\infty {f(ax)e^{-\imath\omega x}} \, {\rm d}x \\
=\frac{1}{a}\frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty {f(x)e^{-\imath \frac{\omega}{a} x}} \, {\rm d}x  = \frac{1}{a}F(f)(\frac{\omega}{a}) \\
(ax = X,dx = \frac{1}{a}dX)
\end{matrix}$$
If a <0  ,then
$$
\begin{matrix}
F(f(ax))(\omega) = \frac{1}{2\pi}\int_{-\infty}^\infty {f(ax)e^{-\imath\omega x}} \, {\rm d}x \\
=\frac{1}{a}\frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty {f(x)e^{-\imath \frac{\omega}{a} x}} \, {\rm d}x  = -\frac{1}{a}F(f)(\frac{\omega}{a}) \\
(ax = X,dx = \frac{1}{a}dX)
\end{matrix}$$
Hence for all a $ \neq 0$,we can write $ F(f(ax))(\omega)= \frac{1}{|a|}F(f)(\frac{\omega}{a})$
(b) We have  
$ F(e^{-|x|})(\omega) = \sqrt{\frac{2}{\pi}}\frac{1}{1 + \omega^2}$
By (a),
$ F(e^{-2|x|})(\omega) = \frac{1}{2}\sqrt{\frac{2}{\pi}}\frac{1}{1 + (\omega/2)^2} = \sqrt{\frac{2}{\pi}}\frac{2}{4 + \omega^2}$

>6

>7
$$
\begin{array}{cccc}
\\
C+ \\
C-
\end{array}
\begin{matrix}
C & G & C & G \\
0.1637&0.2679 * 0.1637&0.3318 * last value&0.2679 * last value\\
0.1267&0.000076 * 0.1267&0.000245 * last value&0.000076 * last value
\end{matrix}$$

>8
Before splitting entropy:
$ Entropy\_0 = -\frac{1}{2}\times log(\frac{1}{2})\times 2 = 1$
Key:x_1
$Entropy\_x_1 = \frac{5}{8}\times (-\frac{3}{5}log(\frac{3}{5}) -\frac{2}{5}log(\frac{2}{5}) ) + \frac{3}{8}\times (-\frac{1}{3}log(\frac{1}{3}) -\frac{2}{3}log(\frac{2}{3}) )  = 0.9512$
$gain\_a = Entropy\_0 -  Entropy\_x_1 = 0.0488$
Key:X_2
$ Entropy\_x_2 =  \frac{1}{2}\times (-\frac{1}{4}log(\frac{1}{4}) -\frac{3}{4}log(\frac{3}{4}) ) + \frac{1}{2}\times (-\frac{1}{4}log(\frac{1}{4}) -\frac{3}{4}log(\frac{3}{4}) ) = 0.811$
$gain\_b = Entropy\_0 -  Entropy\_x_2 = 0.189 $
Key:x_3
$Entropy\_x_3 = \frac{6}{8}\times (-\frac{1}{2}log(\frac{1}{2}) -\frac{1}{2}log(\frac{1}{2}) ) + \frac{2}{8}\times (-\frac{1}{2}log(\frac{1}{2}) -\frac{1}{2}log(\frac{1}{2}) )  = 1$
$gain\_b = Entropy\_0 -  Entropy\_x_3 = 0 $







