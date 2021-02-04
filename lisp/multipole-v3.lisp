(defun I^m_l (m l x y z)
  (let ((r (sqrt (+ (* x x) (* y y) (* z z)))))
    (when (and (= m 0 l 0))
      (return-from I^m_l (/ 1 r)))
    (when (= m l)
      (return-from I^m_l (* (+ m m -1)
			    (/ (complex x y) (* r r))
			    (I^m_l (- m 1) (- m 1) x y z))))
    (when (= l (1+ m))
      (return-from I^m_l (* (+ m m 1)
			    (/ z (* r r))
			    (I^m_l m m x y z))))
    (/ (- (* (+ l l -1) z (I^m_l m (1- l) x y z))
	  (* (- (* (1- l) (1- l)) (* m m )) (I^m_l m (- l 2) x y z)))
       (* r r))))

(defun R^m_l (m l x y z)
  (when (and (= m 0) (= l 0))
    (return-from R^m_l 1))
  (when (= m l)
    (return-from R^m_l
      (* (/ (complex x y) (* 2 m))
	 (R^m_l (1- m) (1- m) x y z))))
  (when (= l (1+ m))
    (return-from R^m_l
      (* z (R^m_l m m x y z))))
  (/ (- (* (- (* 2 l) 1) z (R^m_l m (1- l) x y z))
	(* (+ (* x x) (* y y) (* z z)) (R^m_l m (- l 2) x y z)))
     (- (* l l) (* m m))))

(defun mget (mult n m)
  (when (> (abs m) n)
    (format t "!! mget ~a ~a is invalid~%" n m)
    (return-from mget 0))
  (let* ((c (* n (1+ n)))  ;; center (m=0)
	 (r (+ c m))       ;; right
	 (l (- c m)))      ;; left
    (cond
      ((= m 0)               (aref mult c))
      ((> m 0)   (complex    (aref mult r)     (aref mult l)))
      ((evenp m) (complex    (aref mult l)  (- (aref mult r))))
      (t         (complex (- (aref mult l))    (aref mult r))))))

(defun gen-multipole (p gen-fn)
  (let ((res (make-array (* (1+ p) (1+ p))))
	(off 0))
    (dotimes (n (1+ p))
      (dotimes (m (1+ n))
	(let ((Ynm (funcall gen-fn n m)))
	  (setf (aref res (+ off n (- m))) (imagpart Ynm)
		(aref res (+ off n m)) (realpart Ynm))))
      (incf off (1+ (* 2 n))))
    res))
  
(defun p2m-v3 (dx dy dz q p)
  (gen-multipole p #'(lambda (n m)
		       (* q (R^m_l m n dx dy dz)))))

(defun p2l-v3 (dx dy dz q p)
  (gen-multipole p #'(lambda (n m)
		       (* q (I^m_l m n dx dy dz)))))
    
(defun m2m-v3 (dx dy dz p mult)
  (let ((scratch (p2m-v3 dx dy dz 1 p)))
    (gen-multipole p #'(lambda (n m)
			 (loop for k from 0 to n sum
			      (loop for l from (- k) to k sum
				   (* (mget scratch k l)
				      (mget mult (- n k) (- m l)))))))))

(defun l2l-v3 (dx dy dz p mult)
  (let ((scratch (p2m-v3 dx dy dz 1 p)))
    (gen-multipole p #'(lambda (n m)
			 (loop for k from 0 to (- p n) sum
			      (loop for l from (- k) to k sum
				   (* (conjugate (mget scratch k l))
				      (mget mult (+ n k) (+ m l)))))))))

(defun rot-m-v3 (mult α p)
  (gen-multipole p #'(lambda (n m)
		       (* (mget mult n m) (exp (* #C(0 1.d0) m α))))))

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

(defun gen-b (p)
  (let ((res (make-array (list (1+ (* 2 p)) (1+ (* 2 p))))))
    (loop for l from (- p) to p do
	 (loop for m from (- p) to p do
	      (setf (aref res (+ m p) (+ l p)) (b p m l))))
    res))

(defun gen-sparse-b (n)
  (let* ((dense-b (gen-b n))
	 (res (make-array (array-dimensions dense-b) :initial-element 0)))
    (loop for m from 0 to n do
	 (loop for l from 0 to n do
	      (let* ((a (aref dense-b (+ n m) (+ n l))))
		(cond
		  ((evenp (+ m n))
		   (cond
		     ((= l 0)   (setf (aref res (+ n m) (+ n l)) a))
		     ((evenp l) (setf (aref res (+ n m) (+ n l)) (* 2 a)))
		     (t         (setf (aref res (- n m) (- n l)) (* 2 a)))))
		  (t
		   (if (evenp l)
		       (setf (aref res (- n m) (- n l)) (* 2 a))
		       (setf (aref res (+ n m) (+ n l)) (* 2 a))))
		  ))))
    res))

(defun gen-sparse-bT (n)
  (let* ((dense-b (gen-b n))
	 (res (make-array (array-dimensions dense-b) :initial-element 0)))
    (loop for m from 0 to n do
	 (loop for l from 0 to n do
	      (let* ((a (aref dense-b (+ n l) (+ n m))))
		(cond
		  ((evenp (+ m n))
		   (cond
		     ((= l 0)   (setf (aref res (+ n m) (+ n l)) a))
		     ((evenp l) (setf (aref res (+ n m) (+ n l)) (* 2 a)))
		     (t         (setf (aref res (- n m) (- n l)) (* 2 a)))))
		  (t
		   (if (evenp l)
		       (setf (aref res (- n m) (- n l)) (* 2 a))
		       (setf (aref res (+ n m) (+ n l)) (* 2 a))))
		  ))))
    res))

(defun %swap-xz (mult p fn)
  (loop for i from 1 to p do
       (let* ((v (make-array (1+ (* 2 i))
			     :displaced-to mult
			     :displaced-index-offset (* i i)))
	      (b (gen-sparse-b i))
	      (new-v (funcall fn b v)))
	 (replace v new-v)))
  mult)

(defun mat*vec (a b)
  (let ((res (make-array (array-dimension b 0) :initial-element 0)))
    (dotimes (i (array-dimension b 0))
      (dotimes (j (array-dimension a 1))
	(incf (aref res j) (* (aref a j i) (aref b i)))))
    res))

(defun swapT-xz-v3 (mult p)
  (loop for i from 1 to p do
       (let* ((v (make-array (1+ (* 2 i))
			     :displaced-to mult
			     :displaced-index-offset (* i i)))
	      (b (gen-sparse-bT i))
	      (new-v (mat*vec b v)))
	 (replace v new-v))))

(defun swap-xz-v3 (mult p)
  (loop for i from 1 to p do
       (let* ((v (make-array (1+ (* 2 i))
			     :displaced-to mult
			     :displaced-index-offset (* i i)))
	      (b (gen-sparse-b i))
	      (new-v (mat*vec b v)))
	 (replace v new-v))))

(defun fact (i)
  (if (= i 0) 1 (* i (fact (1- i)))))

(defun m2l-v3 (x y z p mult)
  (let ((αz (atan  x y))
	(αx (- (atan (sqrt (+ (* x x) (* y  y))) z)))
	(m1 (copy-tree mult))
	(r (sqrt (+ (* x x) (* y y) (* z z))))
	(f))
    (format t "αz = ~a, αx = ~a~%" αz αx)
    (setf m1 (rot-m-v3 m1 αz p))
    (swapT-xz-v3 m1 p)
    (setf m1 (rot-m-v3 m1 αx p))
    (swapT-xz-v3 m1 p)
    (format t "before = ~a~%" m1)
    (setf f (gen-multipole p #'(lambda (n m)
				 (loop for k from m to (- p n) sum
				      (progn
					(format t "n+k = ~a~%" (+ n k))
					(/ (* (if (oddp m) -1 1)
					      (mget m1 k m)
					      (fact (+ n k)))
					   (expt r (+ n k 1))))))))
    (format t "f = ~a~%" f)
    (swap-xz-v3 f p)
    (setf f (rot-m-v3 f (- αx) p))
    (swap-xz-v3 f p)
    (setf f (rot-m-v3 f (- αz) p))
    f))

(defun check (p)
  (let* ((mm-v3 (p2m-v3 13.0d0 15.0d0 17.d0 2.0d0 p))
	 (mm (p2m 13.0d0 15.0d0 17.0d0 2.0d0 p))
	 (aa (m2l 5.0d0 6.0d0 7.d0 p mm))
	 (bb (m2l-v3 5.0d0 6.0d0 7.d0 p mm-v3))
	 (aa2 (gen-multipole p #'(lambda (n m)
				   (mult-get aa n m)))))
    (format t "mm = ~a~%" mm)
    (format t "aa = ~a~%" aa2)
    (format t "bb = ~a~%" bb)
  (reduce #'max (mapcar #'abs (map 'list #'- aa2 bb)))))

(defun gen-c-swap-fn (n)
  (let ((b (gen-sparse-b n)))
    (format t "~%")
    (format t "//////////////////////~%")
    (format t "// Swap XZ @ order ~d~%" n)
    (format t "//////////////////////~%")
    (format t "~%")
    (format t "IRIS_CUDA_DEVICE_HOST~%")
    (format t "void swap_xz~d(iris_real *D, iris_real *S) {~%" n)
    (dotimes (i (+ (* 2 n) 1))
      (let ((has-row nil)
	    (first-col t))
	(dotimes (j (+ (* 2 n) 1))
	  (let ((a (aref b i j)))
	    (when (/= a 0)
	      (if first-col
		  (format t "  D[~a] = " i)
		  (format t " + "))
	      (format t "S[~a] * ~,,,,,,'eg" j (* a 1.0d0))
	      (setf first-col nil
		    has-row t))))
	(when has-row
	  (format t ";~%"))))
    (format t "}~%")))

(defun gen-c-swapT-fn (n)
  (let ((b (gen-sparse-bT n)))
    (format t "~%")
    (format t "//////////////////////~%")
    (format t "// SwapT XZ @ order ~d~%" n)
    (format t "//////////////////////~%")
    (format t "~%")
    (format t "IRIS_CUDA_DEVICE_HOST~%")
    (format t "void swapT_xz~d(iris_real *D, iris_real *S) {~%" n)
    (dotimes (i (+ (* 2 n) 1))
      (let ((has-row nil)
	    (first-col t))
	(dotimes (j (+ (* 2 n) 1))
	  (let ((a (aref b i j)))
	    (when (/= a 0)
	      (if first-col
		  (format t "  D[~a] = " i)
		  (format t " + "))
	      (format t "S[~a] * ~,,,,,,'eg" j (* a 1.0d0))
	      (setf first-col nil
		    has-row t))))
	(when has-row
	  (format t ";~%"))))
    (format t "}~%")))

(defun gen-all-c-swap-fn ()
  (loop for i from 1 to 20 do
    (gen-c-swap-fn i)
    (gen-c-swapT-fn i)))
  
