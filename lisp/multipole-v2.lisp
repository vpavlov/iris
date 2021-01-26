(defun mult-idx (n m)  ;; n is lower index [0..p], m is upper index [-n..n]
  (assert (<= (abs m) n))
  (+ (* n n) n m))

(defun mult-get (mult n m)
  (when (> (abs m) n)
    (return-from mult-get 0))
  (nth (mult-idx n m) mult))

;; Forward expansion

(defun υnm (n m x y z)  ;; real
  (when (and (= n 0) (= m 0))
    (return-from υnm 1))
  (when (= n m)
    (return-from υnm
      (/ (- (* x (υnm (1- m) (1- m) x y z))
	    (* y (τnm (1- m) (1- m) x y z)))
	 (* 2 m))))
  (when (= n (1+ m))
    (return-from υnm
      (* z (υnm m m x y z))))
  (/ (- (* (- (* 2 n) 1) z (υnm (1- n) m x y z))
	(* (+ (* x x) (* y y) (* z z)) (υnm (- n 2) m x y z)))
     (- (* n n) (* m m))))

(defun τnm (n m x y z) ;; imag
  (when (and (= n 0) (= m 0))
    (return-from τnm 0))
  (when (= n m)
    (return-from τnm
      (/ (+ (* x (τnm (1- m) (1- m) x y z))
	    (* y (υnm (1- m) (1- m) x y z)))
	 (* 2 m))))
  (when (= n (1+ m))
    (return-from τnm
      (* z (τnm m m x y z))))
  (/ (- (* (- (* 2 n) 1) z (τnm (1- n) m x y z))
	(* (+ (* x x) (* y y) (* z z)) (τnm (- n 2) m x y z)))
     (- (* n n) (* m m))))

(defun fact (i)
  (if (= i 0) 1 (* i (fact (1- i)))))

(defun unm (n m x y z)
  (let ((f (* (fact (- n m)) (fact (+ n m)))))
    (if (>= m 0)
	(* 1 (υnm n m x y z))
	(* 1 (τnm n (- m) x y z)))))


;; Backward expansion

(defun ζnm (n m x y z)  ;; real
  (let ((r (sqrt (+ (* x x) (* y y) (* z z)))))
    (when (and (= m 0 n 0))
      (return-from ζnm (/ 1 r)))
    (when (= m n)
      (return-from ζnm (* (/ (+ m m -1) (* r r))
			  (- (* x (ζnm (1- m) (1- m) x y z))
			     (* y (χnm (1- m) (1- m) x y z))))))
    (when (= n (1+ m))
      (return-from ζnm (* (+ m m 1)
			  (/ z (* r r))
			  (ζnm m m x y z))))
    (/ (- (* (+ n n -1) z (ζnm (1- n) m x y z))
	  (* (- (* (1- n) (1- n)) (* m m )) (ζnm (- n 2) m x y z)))
       (* r r))))

(defun χnm (n m x y z)  ;; real
  (let ((r (sqrt (+ (* x x) (* y y) (* z z)))))
    (when (and (= m 0 n 0))
      (return-from χnm 0))
    (when (= m n)
      (return-from χnm (* (/ (+ m m -1) (* r r))
			  (+ (* x (χnm (1- m) (1- m) x y z))
			     (* y (ζnm (1- m) (1- m) x y z))))))
    (when (= n (1+ m))
      (return-from χnm (* (+ m m 1)
			  (/ z (* r r))
			  (χnm m m x y z))))
    (/ (- (* (+ n n -1) z (χnm (1- n) m x y z))
	  (* (- (* (1- n) (1- n)) (* m m )) (χnm (- n 2) m x y z)))
       (* r r))))

(defun tnm (n m x y z)
  (let* ((r (sqrt (+ (* x x) (* y y) (* z z))))
	 (f (expt r (+ n n 1))))
    (if (>= m 0)
	(* 1 (ζnm n m x y z))
	(* 1 (χnm n (- m) x y z)))))


(defun p2m (dx dy dz q p)
  (mapcan #'identity (loop for n from 0 to p collect
			  (loop for m from (- n) to n collect
			       (* q (if (= m 0)
					(unm n m dx dy dz)
					(+ (unm n m dx dy dz)
					   (* #C(0 1) (unm n (- m) dx dy dz)))))))))


(defun p2l (dx dy dz q p)
  (mapcan #'identity (loop for n from 0 to p collect
			  (loop for m from (- n) to n collect
			       (* q (tnm n m dx dy dz))))))

(defun m2m (dx dy dz p mult)
  (let ((scratch (p2m dx dy dz 1 p)))
    (mapcan
     #'identity
     (loop for n from 0 to p collect
	  (loop for m from (- n) to n collect
	       (loop for k from 0 to n sum
		    (loop for l from (- k) to k sum
			 (* (mult-get scratch k l)
			    (mult-get mult (- n k) (- m l))))))))))


(defun m2l (dx dy dz p mult)
  (let ((tmp (p2l dx dy dz 1 p)))
    (mapcan
     #'identity
     (loop for n from 0 to p collect
	  (loop for m from (- n) to n collect
	       (loop for k from 0 to (- p n) sum
		    (loop for l from (- k) to k sum
			 (* (mult-get mult k l)
			    (mult-get tmp (+ n k) (+ m l))))))))))

;; Eq. 3e
(defun l2l (dx dy dz p loc)
  (let ((tmp (p2m dx dy dz 1 p)))
    (mapcan
     #'identity
     (loop for n from 0 to p collect
	  (loop for m from (- n) to n collect
	       (loop for k from 0 to (- p n) sum
		    (loop for l from (- k) to k sum
			 (* (mult-get tmp k l)
			    (mult-get loc (+ n k) (+ m l))))))))))


;; (p2p 0.5 0.5 0.5 -1 12 12 12 1)
(defun p2p (x1 y1 z1 q1 x2 y2 z2 q2)
  (let* ((dx (- x2 x1))
	 (dy (- y2 y1))
	 (dz (- z2 z1))
	 (ri^2 (+ (* dx dx) (* dy dy) (* dz dz)))
	 (ri (sqrt ri^2))
	 (e (/ (* q1 q2) ri))
	 (ee (/ e ri^2))
	 (fx (* ee dx))
	 (fy (* ee dy))
	 (fz (* ee dz)))
    (format t "φ = ~a, f = (~a, ~a, ~a)~%" (/ q1 ri) fx fy fz)))

;; (e2e 10)
(defun e2e (p)
  (let ((res (l2l -0.14135 -0.14135 -0.14135 p
		  (l2l 0.93395 0.93395 0.93395 p
		       (m2l 9.3395 9.3395 9.3395 p
			    (m2m -1.8679 -1.8679 -1.8679 p
				 (m2m -0.93395 -0.93395 -0.93395 p
				      (p2m -0.43395 -0.43395 -0.43395 -1 p))))))))
    (format t "φ = ~a, f = (~a ~a ~a)~%" (first res)
	    (fourth res)
	    (second res)
	    (third res))))


(defun b (n m l)
  (if (and (= n 0) (= m 0) (= l 0))
      (progn
	(return-from b 1))
      (if (> (abs l) n)
	  (return-from b 0)
	  (if (<= (abs m) (1- n))
	      (let ((res (/ (- (b (1- n) m (1- l))
			       (b (1- n) m (1+ l))) 2)))
		(return-from b res))
	      (if (> m 0)
		  (let ((res (/ (+ (b (1- n) (1- m) (1- l))
				   (b (1- n) (1- m) (1+ l))
				   (* 2 (b (1- n) (1- m) l))) 2)))
		    (return-from b res))
		  (let ((res (/ (+ (b (1- n) (1+ m) (1- l))
				   (b (1- n) (1+ m) (1+ l))
				   (* -2 (b (1- n) (1+ m) l))) 2)))
		    (return-from b res)))))))

(defun gen-b (p transp)
  (let ((res (make-array (list (1+ (* 2 p)) (1+ (* 2 p)))))
	(first1 t)
	(first2 t))
    (loop for l from (- p) to p do
	 (loop for m from (- p) to p do
	      (setf (aref res (+ m p) (+ l p)) (* 1.0d0 (b p m l)))))
    (format t "iris_real ~a~d[][~d] = {~%" (if transp "bt" "b") p (1+ (* 2 p)))
    (dotimes (i (1+ (* 2 p)))
      (setf first2 t)
      (when (not first1)
	(format t ",~%"))
      (setf first1 nil)
      (format t "    { ")
      (dotimes (j (1+ (* 2 p)))
	(when (not first2)
	  (format t ", "))
	(setf first2 nil)
	(format t "~,,,,,,'eg" (if transp (aref res j i) (aref res i j))))
      (format t " }"))
    (format t "~%};~%")
    (values)))

(defun gen-all-b ()
  (dotimes (i 21)
    (format t "~%////////////////~%")
    (format t "// B~a matrix //~%" i)
    (format t "////////////////~%~%")
    (gen-b i nil)
    (format t "~%/////////////////~%")
    (format t "// BT~a matrix //~%" i)
    (format t "/////////////////~%~%")
    (gen-b i t)))
  
