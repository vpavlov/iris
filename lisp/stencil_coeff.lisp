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
  (when (> k (/ n 2))
    (setf k (- n k)))
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
  - for p = 4: { 1, -1/12, 1/90, -1/560 } * [cdo^2, cdo^4, cdo^6]
"
  (let ((retval (make-array p))
	(sign 1))
    (loop for k from 0 below p do
	 (let ((kfact (fact k)))
	   (setf (aref retval k) (/ (* sign kfact kfact)
				    (* (1+ k) (fact (1+ (* 2 k)))))))
	 (setf sign (* -1 sign)))
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
	   
