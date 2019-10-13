
;; FINITE DIFFERENCE OPERATORS
;;
;; Identity:            1 y[n] = y[n]
;; Forward difference:  Δ y[n] = y[n+1] - y[n]
;; Backward difference: Γ y[n] = y[n] - y[n+1]
;; Central difference:  δ y[n] = y[n+1/2] - y[n-1/2]
;; Averaging:           μ y[n] = 1/2 * ( y[n+1/2] - y[n-1/2 )
;; Shift:               E y[n] = y[n+1]
;; Integral:            J y    = Definite integral from x to x+h y(t) dt
;; Differential:        D y    = dy/dx
;;
;; Some useful relations:
;;
;; Δ = E - 1
;; DJ = JD = Δ
;; Γ = 1 - E^-1
;; μ = 1/2 * (Ε^1/2 + Ε^-1/2)
;; δ = E^1/2 - E^-1/2
;; μ^2 = 1 + 1/4 δ^2
;;
;; To proceed, the operator D must be related to the other operators. For
;; this purpose the Taylor series
;;
;; f(x+h)  = f(x) + h/1! f'(x) + h^2/2! f''(x) + ...
;;
;; can be expressed in the operational form
;;
;; Ef(x) = [ 1 + hD/1! + h^2*D^2/2! + ... ] = e^(hD) * f(x)
;; 
;; thus E = e^(hD), from where it follows that:
;;
;; hD = ln(E) = ln(1+Δ) = -ln(1-Γ)
;;
;;
;; hD = ln(E) = 2 * ln(E^1/2) = 2*ln(1/2*(E^1/2 + E^-1/2)+1/2*(E^1/2 - E^-1/2))
;;    = 2 * ln(μ + 1/2 δ) = 2*ln(1/2 δ + sqrt(1 + (1/2δ)^2)) = 2*asinh(δ/2)
;;
;; hD = 2*asinh(δ/2)
;;
;; D = 1/h * 2*asinh(δ/2)
;; D^2 = 1/h^2 * 4*asinh(δ/2)^2


;; The whole purpose of this file is to calculate the coefficients of a n-th
;; order stencil. We do this in lisp, since it supports fractional numbers and
;; big integers, so factorial is not a problem.
;;
;; Usage: load/compile everything, then call
;;
;; (stencil-coeff N)
;;
;; where N is the order of the stencil (must be even). This is the power of h
;; (mesh step) to which the taylor expansion is correct
;;
;; (stencil-coeff 2) produces #(1 -2 1), which is the central difference
;; operator of second order (in 1D):
;;   1 * u[x+1] -2 * u[x] + 1 * u[x-1]
;;
;; (stencil-coeff 4) produces 4th order stencil in 1D:
;; -1/12 * (u[x+2] -16*u[x+1] +30*u[x] -16*u[x-1] + u[x-2])
;;
;; and so on.
;; 
;; In order to do this for 2D and 3D, just multiply the central coefficient *
;; 2, respectively 3.


(defun binom (n k)
  "Calculate the binomial coefficient (n k) using the multiplicative formula"
  ;; (when (> k (/ n 2))
  ;;   (setf k (- n k)))
  (let ((retval 1))
    (loop for i from 1 to k do
	 (setf retval (* retval (/ (- (1+ n) i) i))))
    retval))

(defun fact (n)
  (if (= n 0) 1 (* n (fact (1- n)))))

(defun cdo-coeff (n)
  "Return the central difference operator of order n (δ^n) coefficients 

The coefficients are n+1 and follow the formula: kth coeff is (-1)^k (n k)

For example:
  for n = 1 coefficients are { 1, -1 }
  for n = 2 coefficients are { 1, -2, 1 }
  for n = 3 coefficients are { 1, -3, 3, -1 }
  for n = 4 coefficients are { 1, -4, 6, -4, 1 }, etc.
"
  (let ((retval (make-array (1+ n)))
	(sign 1))
    (loop for k from 0 to n do
	 (setf (aref retval k) (* sign (binom n k)))
	 (setf sign (* sign -1)))
    retval))


(defun d2/dx2-taylor-coeff (p)
  "Taylor Expansion d^2/dx^2 ~ 4*asinh^2(x/2) ~
  sum [ (-1)^k * (x^(2+2k) (k!)^2) / (1+k) (1+2k)! ]

p = number of terms in the expansion

For example:
  - for p = 1: 1 * [cdo^2]
  - for p = 2: { 1, -1/12 } * [cdo^2, cdo^4]
  - for p = 3: { 1, -1/12, 1/90 } * [cdo^2, cdo^4, cdo^6]
  - for p = 4: { 1, -1/12, 1/90, -1/560 } * [cdo^2, cdo^4, cdo^6, cdo^8]
"
  (let ((retval (make-array p))
	(sign 1))
    (loop for k from 0 below p do
	 (let ((kfact (fact k)))
	   (setf (aref retval k) (/ (* sign kfact kfact)
				    (* (1+ k) (fact (1+ (* 2 k)))))))
	 (setf sign (* -1 sign)))
    retval))

(defun d/dx-taylor-coeff (p)
  "Taylor Expansion d/dx ~ 2*asinh(δ/2) ~
  2 * sum [ [ (-1)^k / 2^(2k+1) * (2k + 1) ] * (k - 1/2 | k) * δ^(2k+1) ] 

p = number of terms in the expansion

For example:
  - for p = 1: 1 * [δ]
  - for p = 2: { 1, -1/24 } * [δ, δ^3]
  - for p = 3: { 1, -1/24, 3/640 } * [δ, δ^3, δ^5]
"
  (let ((retval (make-array p))
	(sign 1)
	(ptwo 2))
    (dotimes (k p)
      (let ((a (binom (- k 1/2) k)))
	(setf (aref retval k) (/ (* 2 sign a) (* ptwo (1+ (* 2 k))))
	      sign (* -1 sign)
	      ptwo (* ptwo 4))))
    retval))

(defun stencil-coeff(order)
  (let* ((retval (make-array (1+ order)))
	 (p (/ order 2))
	 (a (d2/dx2-taylor-coeff p))
	 (offset 0))
    (loop for i from (1- p) downto 0 do
	 (let* ((r (* (1+ i) 2))
		(rth (cdo-coeff r)))
	   (loop for j from 0 below (1+ r) do
		(incf (aref retval (+ j offset))
		      (* (aref rth j) (aref a i)))))
	 (incf offset))
    retval))
	   

(defun %pade-A (c m n)
  "Find the matrix that will solve the linear system of equations to find the
Pade [M, N] approximation, given the Taylor expansion coefficients C.

We start with a Tyalor expansion, given through the coefficients C:

c0 + c1*x + c2*x^2 +... = (a0 + a1*x + a2*x^2 +...) / (1 + b1*x + b2*x^2+...)

M is the number of a's and N is the number of b's required.

Multiplying through the denominator we get a system of equations

a0 = c0
a1 = c1 + c0 * b1
a2 = c2 + c1 * b1 + c0 * b2
...

We cut off the system to M+N+1 equations of M+N+1 unknowns (a's and b's).

A complete example:

Start with a taylor expansion of d2/dx2 up to second order:

 (d2/dx2-taylor-coeff 2)

gives back 

 #(1 -1/12)

This means that d2/dx2 =~ δ^2/h^2 * (1 - 1/12 δ^2). We deal only with the 
expression in the brackets, which is

1*δ^0 + 0*δ^1 - (1/12)*δ^2,

so C is (list 1 0 -1/12)

Since the highest degree is 2, M+N must = 2. The case of N = 0 is trivial, since
it is just equal to the Taylor expansion. Let's try M = 0, N = 2. For this case
we have

1*δ^0 + 0*δ^1 - (1/12)*δ^2 = (a0 + a1*δ + a2*δ^2) / (1 + b1*δ + b2*δ^2)

a0 = c0
 0 = c1 + c0*b1
 0 = c2 + c1*b1 + c0*b2

or in matrix form

|1  0  0| * |a0| = | c0|
|0 c0  0|   |b1|   |-c1|
|0 c1 c0|   |b2|   |-c2|

|1 0 0| * |a0| = |   1|
|0 1 0|   |b1| = |   0|
|0 0 1|   |b2| = |1/12|

So, according to the Pade approximation P[0,2],

d2/dx2 =~ δ^2/h^2 * 1 / (1 + 1/12 δ^2)


This function returns the square matrix for the left hand side.

The matrix is M+N+1 square and is constructed as follows:

1. Starting from the top-left, we put (M+1)x(M+1) identity matrix
2. Directly below, put Nx(M+1) zero matrix. With this, we have a M+N+1 rows
   of M+1 columns. What's left is to add to the right N columns.
3. The i-th such column (starting from 1) starts with i zeros and continues
   with c0, c1, c2, etc. up till there are M+N+1 elements in the column. If the
   corresponding row is in the first M+1 rows, add minus sign.

E.g. for a [2, 2] we need to make 2+2+1 = 5x5 square matrix A:

A = |. . . . .|, then after step 1: A = |1 0 0 . .|, then after step 2:
    |. . . . .|                         |0 1 0 . .|
    |. . . . .|                         |0 0 1 . .|
    |. . . . .|                         |. . . . .|
    |. . . . .|                         |. . . . .|

A = |1 0 0 . .|, then, after Step 3 first column: A = |1 0 0   0 .|
    |0 1 0 . .|                                       |0 1 0 -c0 .|
    |0 0 1 . .|                                       |0 0 1 -c1 .|
    |0 0 0 . .|                                       |0 0 0  c2 .|
    |0 0 0 . .|                                       |0 0 0  c3 .|

and finally A = |1 0 0   0   0|
		|0 1 0 -c0   0|
		|0 0 1 -c1 -c0|
		|0 0 0  c2  c1|
		|0 0 0  c3  c2|"
  (let ((a (make-array (list (+ m n 1) (+ m n 1)) :initial-element 0)))

    ;; Step 1
    (loop for i from 0 to m do
	 (setf (aref a i i) 1))

    ;; Step 2 is taken care from the initialization above
    
    ;; Step 3
    (loop for i from (1+ m) to (+ m n) do
	 (loop for j from (- i m) to (+ m n) do
	      (let ((val (aref c (- j (- i m)))))
		(when (<= j m)
		  (setf val (- val)))
		(setf (aref a j i) val))))
	      
    a))

(defun %pade-B (c m n)
  "Right hand side of the same (see %pade-A) is build as follows:

Get all the c's one after the other, the first m+1 of them being with a positive
sign, the rest with negative"
  (let ((b (make-array (+ m n 1) :initial-element 0)))
    (loop for i from 0 to (+ m n) do
	 (let ((val (aref c i)))
	   (when (> i m)
	     (setf val (- val)))
	   (setf (aref b i) val)))
    b))

(defun solve-linsys (a b)
  "Gaussian elimination. Nothing fancy, no pivoting, etc. Should be enough for
the type of systems we're solving here"
  (let* ((ad (array-dimensions a))
	 (adb (array-dimensions b))
	 (n (first ad)))

    (when (or (/= (length ad) 2)
	      (/= (length adb) 1)
	      (/= n (second ad))
	      (/= n (first adb)))
      (error "Invalid arguments!"))

    (let ((x (make-array n :initial-element 0)))

      ;; Gaussian elimination
      (loop for k from 0 below (1- n) do
	   (loop for i from (1+ k) below n do
		(setf (aref a i k) (/ (aref a i k) (aref a k k)))
		(loop for j from (1+ k) below n do
		     (decf (aref a i j) (* (aref a i k) (aref a k j))))))
      
      ;; Forward elimination
      (loop for k from 0 below (1- n) do
	   (loop for i from (1+ k) below n do
		(decf (aref b i) (* (aref a i k) (aref b k)))))
      
      ;; Backward solve
      (loop for i from (1- n) downto 0 do
	   (let ((s (aref b i)))
	     (loop for j from (1+ i) below n do
		  (decf s (* (aref a i j) (aref x j))))
	     (setf (aref x i) (/ s (aref a i i)))))
      x)))
					   
(defun pade (c m n)
  (let ((a (%pade-A c m n))
	(b (%pade-B c m n)))
    (solve-linsys a b)))

(defun pade-stencil (m n &optional cut)
  (let* ((order (1+ (/ (+ m n) 2)))
	 (c (make-array (+ m n 1) :initial-element 0))
	 (tt (d2/dx2-taylor-coeff order)))
    (dotimes (i (length tt))
      (setf (aref c (* i 2)) (aref tt i)))
    (let* ((pc (pade c m n))
	   (nom (remove-if #'zerop (coerce (subseq pc 0 (1+ m)) 'list)))
	   (denom (remove-if #'zerop
			     (cons 1 (coerce (subseq pc (1+ m)) 'list)))))
      (multiple-value-bind (rhs-mult rhs-stencil)
	  (rhs-stencil (+ m n 2) denom cut)
	(format t "1/~a ~a~%" rhs-mult rhs-stencil)
	(multiple-value-bind (lhs-mult lhs-stencil)
	    (lhs-stencil (+ m n 2) nom denom cut)
	  (values lhs-stencil (/ lhs-mult rhs-mult) rhs-stencil))))))

(defun rhs-stencil (m poly cut)
  (let* ((n (length poly))
	 (res (cdo m 0 0 0 0)))  ;; empty stencil
    (loop for i from 0 below n do
	 (loop for j from 0 below n do
	      (loop for k from 0 below n do
		 (when (or (not cut) (<= (+ (* i 2) (* j 2) (* k 2)) (+ m 0)))
		     (setf res (cdo+ res 
				     (cdo m (* (nth i poly) (nth j poly)
					       (nth k poly))
					  (* i 2) (* j 2) (* k 2))))))))
    ;; extract common denominator
    (extract-denom-lcm-3d res)))

(defun lhs-stencil (m nom denom cut)
  (let* ((n1 (length nom))
	 (n2 (length denom))
	 (res (cdo m 0 0 0 0))) ;; empty stencil

    ;; nom x * denom y * denom z
    (loop for i from 0 below n1 do
	 (loop for j from 0 below n2 do
	      (loop for k from 0 below n2 do
		 (when (or (not cut) (<= (+ (* (1+ i) 2) (* j 2) (* k 2)) (+ m 0)))
		     (setf res (cdo+ res
				     (cdo m (* (nth i nom)
					       (nth j denom)
					       (nth k denom))
					  (* (1+ i) 2) (* j 2) (* k 2))))))))

    ;; denom x * nom y * denom z
    (loop for i from 0 below n2 do
	 (loop for j from 0 below n1 do
	      (loop for k from 0 below n2 do
		 (when (or (not cut) (<= (+ (* i 2) (* (1+ j) 2) (* k 2)) (+ m 0)))
		     (setf res (cdo+ res
				     (cdo m (* (nth i denom)
					       (nth j nom)
					       (nth k denom))
					  (* i 2) (* (1+ j) 2) (* k 2))))))))

    ;; denom x * denom y * nom z
    (loop for i from 0 below n2 do
	 (loop for j from 0 below n2 do
	      (loop for k from 0 below n1 do
		 (when (or (not cut) (<= (+ (* i 2) (* j 2) (* (1+ k) 2)) (+ m 0)))
		     (setf res (cdo+ res
				     (cdo m (* (nth i denom)
					       (nth j denom)
					       (nth k nom))
					  (* i 2) (* j 2) (* (1+ k) 2))))))))
    (extract-denom-lcm-3d res)))
	   
(defun cdo (n c xp yp zp)
  "Return a central difference operator 3D stencil of accuracy order N (must be
even(?) with a constant coefficient C, and X, Y and Z powers of δ being XP, YP
and ZP respectively. Thus,

 (cdo 2 144 0 0 0)

Will return a 27-point stencil with a central element 144

 (cdo 2 144 2 0 0)

Will return a 27-point stencil for δ^2x, while

 (cdo 2 144 2 2 2)

Will return a 27-point stencil for δ^2x * δ^2y * δ^2z"

  (let* ((retval (make-array (list (1+ n) (1+ n) (1+ n)) :initial-element 0))
	 (center (/ n 2))
	 (x-cdo (cdo-coeff xp))
	 (y-cdo (cdo-coeff yp))
	 (z-cdo (cdo-coeff zp)))
    (let* ((sx (- center (/ (1- (length x-cdo)) 2)))
	   (ex (+ sx (1- (length x-cdo)))))
      (loop for x from sx to ex do
	   (let* ((v1 (* c (aref x-cdo (- x sx))))
		  (sy (- center (/ (1- (length y-cdo)) 2)))
		  (ey (+ sy (1- (length y-cdo)))))
	     (loop for y from sy to ey do
		  (let* ((v2 (* v1 (aref y-cdo (- y sy))))
			 (sz (- center (/ (1- (length z-cdo)) 2)))
			 (ez (+ sz (1- (length z-cdo)))))
		    (loop for z from sz to ez do
			 (setf (aref retval x y z)
			       (* v2 (aref z-cdo (- z sz))))))))))
    retval))

(defun cdo/ (cdo lam)
  (let* ((ad (array-dimensions cdo))
	 (retval (make-array ad :initial-element 0)))
    (dotimes (i (first ad))
      (dotimes (j (second ad))
	(dotimes (k (third ad))
	  (setf (aref retval i j k)
		(/ (aref cdo i j k) lam)))))
    retval))
  
(defun cdo+ (&rest terms)
  (let* ((ad (array-dimensions (first terms)))
	 (retval (make-array ad :initial-element 0)))
    (dotimes (i (first ad))
      (dotimes (j (second ad))
	(dotimes (k (third ad))
	  (setf (aref retval i j k)
		(reduce #'+ (mapcar #'(lambda (term)
					(aref term i j k))
				    terms))))))
    retval))

(defun extract-denom-lcm-3d (stencil)
  "Find the least common multiple of all the denominators, multiply everything
by it and return the modified stencil"
  (let ((ad (array-dimensions stencil))
	(lst ())
	(lcm))

    (dotimes (i (first ad))
      (dotimes (j (second ad))
	(dotimes (k (third ad))
	  (push (aref stencil i j k) lst))))

    (setf lcm (reduce #'lcm (mapcar #'denominator lst)))

    (dotimes (i (first ad))
      (dotimes (j (second ad))
	(dotimes (k (third ad))
	  (setf (aref stencil i j k) (* (aref stencil i j k) lcm)))))
    (values lcm stencil)))

(defun v (k)
  " V(k) = (k!)^2/(1+2k)! without factorials (so no bigints involved)

V(k) = (k!)^2 / (1+2k)*(2k)! = W(k)/(1+2k), W(k) = (k!)^2/(2k)!

W(1) = 1/2
W(k) / W(k-1) = 1 / a(k)

W(k) = W(k-1) / a(k)

a(k) = 3 + b(k)

b(k) = n(k)/d(k)

n(2) = 0
d(2) = 1

n(k) for odd k = 2*n(k-1) + 1, for even k = n(k-2) + 1
d(k) for odd k = 2*d(k-1) + 1, for even k = d(k-2) + 1

 
")

(defun val (k init)
  (when (= k 2)
    (return-from val init))
  (if (oddp k)
      (1+ (* 2 (val (1- k) init)))
      (1+ (val (- k 2) init))))

(defun n (k) (val k 0))
(defun d (k) (val k 1))
(defun b (k) (/ (n k) (d k)))
(defun a (k) (+ 3 (b k)))

(defun w (k)
  (when (= k 1)
    (return-from w 1/2))
  (/ (w (1- k)) (a k)))

(defun v (k)
  (/ (w k) (1+ k) (1+ (* 2 k))))


(defun sinx (n)
  (let ((sinx (make-array n))
	(h (/ (* 2 pi) n)))
    (dotimes (i n)
      (let ((x (* i h)))
	(setf (aref sinx i) (sin x))))
    sinx))

(defun der1 (f)
  (let* ((n (length f))
	 (h (/ (* 2 pi) n))
	 (der (make-array n)))
    (dotimes (i n)
      (let ((i-3 (- i 3))
	    (i-1 (- i 1))
	    (i+1 (+ i 1))
	    (i+3 (+ i 3)))
	(when (< i-3 0) (incf i-3 n))
	(when (< i-1 0) (incf i-1 n))
	(when (> i+3 (1- n)) (decf i+3 n))
	(when (> i+1 (1- n)) (decf i+1 n))
	(setf (aref der i)
	      (/ (+ (* -1 (aref f i+3))
		    (* 27 (aref f i+1))
		    (* -27 (aref f i-1))
		    (* 1 (aref f i-3)))
		 (* 48 h)))))
    der))

(defun der2 (f)
  (let* ((n (length f))
	 (h (/ (* 2 pi) n))
	 (der (make-array n)))
    (dotimes (i n)
      (let ((i-2 (- i 2))
	    (i-1 (- i 1))
	    (i+1 (+ i 1))
	    (i+2 (+ i 2)))
	(when (< i-2 0) (incf i-2 n))
	(when (< i-1 0) (incf i-1 n))
	(when (> i+2 (1- n)) (decf i+2 n))
	(when (> i+1 (1- n)) (decf i+1 n))
	(let ((δ (- (aref f i+1) (aref f i-1)))
	      (δ2 (+ (aref f i+2) (aref f i-2) (* -2 (aref f i)))))
	  (setf (aref der i) (/ (* δ (- 1 (* 1/24 δ2))) (* 2 h))))))
    der))

(defun err (g)
  (let* ((n (length g))
	 (h (/ (* 2 pi) n))
	 (mse 0.0))
    (dotimes (i n)
      (let* ((x (* i h))
	     (e (- (cos x) (aref g i))))
	(incf mse (* e e))))
    (sqrt mse)))
