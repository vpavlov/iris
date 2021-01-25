(defun mult-idx (l m)
  (+ (/ (* l (1+ l)) 2) m))

(defun mult-get (mult l m)
  (when (> (abs m) l)
    (return-from mult-get 0))
  (if (< m 0)
      (* (conjugate (nth (mult-idx l (- m)) mult)) (expt -1 m))
      (nth (mult-idx l m) mult)))

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

;; Eq. 3c
(defun p2m (dx dy dz q p)
  (mapcan #'identity (loop for n from 0 to p collect
			  (loop for m from 0 to n collect
			       (* q (R^m_l m n dx dy dz))))))

(defun p2l (dx dy dz q p)
  (mapcan #'identity (loop for n from 0 to p collect
			  (loop for m from 0 to n collect
			       (* q (I^m_l m n dx dy dz))))))

;; Eq. 3d
(defun m2m (dx dy dz p mult)
  (let ((ips (p2m dx dy dz 1.0d0 p)))
    (mapcan
     #'identity
     (loop for n from 0 to p collect
	  (loop for m from 0 to n collect
	       (loop for k from 0 to n sum
		    (loop for l from (- k) to k sum
			 (* (mult-get ips k l)
			    (mult-get mult (- n k) (- m l))))))))))

	    
  
;; Eq. 3b
(defun m2l (dx dy dz p mult)
  (let ((tmp (p2l dx dy dz 1.0d0 p)))
    (mapcan
     #'identity
     (loop for n from 0 to p collect
	  (loop for m from 0 to n collect
	       (loop for k from 0 to (- p n) sum
		    (loop for l from (- k) to k sum
			 (* (conjugate (mult-get mult k l))
			    (mult-get tmp (+ n k) (+ m l))))))))))
     

;; Eq. 3e
(defun l2l (dx dy dz p loc)
  (let ((tmp (p2m dx dy dz 1.0d0 p)))
    (mapcan
     #'identity
     (loop for n from 0 to p collect
	  (loop for m from 0 to n collect
	       (loop for k from 0 to (- p n) sum
		    (loop for l from (- k) to k sum
			 (* (conjugate (mult-get tmp k l))
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
    (format t "φ = ~a, f = (~a ~a ~a)~%"
	    (realpart (first res))
	    (realpart (third res))
	    (imagpart (third res))
	    (realpart (second res)))))

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

(defun rot-m (mult α p)
  (loop for n from 0 to p do
       (loop for m from 1 to n do ;; 
	    (let ((idx (mult-idx n m)))
	      (setf (nth idx mult) (* (nth idx mult) (exp (* #C(0 1.d0) m α)))))))
  mult)

(defun mult-to-vect (m p)
  (let ((res (make-array (+ (* 2 p) 1))))
    (loop for i from (- p) to p do
	 (setf (aref res (+ i p)) (mult-get m p i)))
    res))

(defun mat*vec (a b)
  (let ((res (make-array (array-dimension b 0) :initial-element 0)))
    (dotimes (i (array-dimension b 0))
      (dotimes (j (array-dimension a 1))
	(incf (aref res j) (* (aref a j i) (aref b i)))))
    res))

(defun matT*vec (a b)
  (let ((res (make-array (array-dimension b 0) :initial-element 0)))
    (dotimes (i (array-dimension b 0))
      (dotimes (j (array-dimension a 1))
	(incf (aref res j) (* (aref a i j) (aref b i)))))
    res))

(defun swap-xz (mult p)
  (loop for i from 0 to p do
       (let* ((v (mult-to-vect mult i))
	      (b (gen-b i))
	      (new-m (mat*vec b v))
	      (nn i))
	 (loop for j from 0 to i do ;; 
	      (setf (nth (mult-idx i j) mult) (aref new-m nn))
	      (incf nn))))
  mult)

(defun swapT-xz (mult p)
  (loop for i from 0 to p do
       (let* ((v (mult-to-vect mult i))
	      (b (gen-b i))
	      (new-m (matT*vec b v))
	      (nn i))
	 (loop for j from 0 to i do ;; 
	      (setf (nth (mult-idx i j) mult) (aref new-m nn))
	      (incf nn))))
  mult)

(defun fact (i)
  (if (= i 0) 1 (* i (fact (1- i)))))

(defun m2l-v2 (x y z p mult)
  (let ((αz (atan  x y))
	(αx (- (atan (sqrt (+ (* x x) (* y  y))) z)))
	(m1 (copy-tree mult))
	(r (sqrt (+ (* x x) (* y y) (* z z))))
	(f))
    (format t "αz = ~a, αx = ~a~%" αz αx)
    (rot-m m1 αz p)
    (swapT-xz m1 p)
    (rot-m m1 αx p)
    (swapT-xz m1 p)
    (setf f (mapcan #'identity 
		    (loop for n from 0 to p collect
			 (loop for m from 0 to n collect
			      (loop for k from m to (- p n) sum
				   (progn
				     (format t "n m k ~a ~a ~a~%" n m k)
				     (/ (* (expt -1 m) (mult-get m1 k m)
					   (fact (+ n k)))
					(expt r (+ n k 1)))))))))
    (format t "ff = ~a~%" f)
    ;;(format t "F[~a,~a] = ~a~%" 6 -6 (mult-get f 6 -6))
    (swap-xz f p)
    (rot-m f (- αx) p)
    (swap-xz f p)
    (rot-m f (- αz) p)
    f))

(defun check (p)
  (let* ((mm (p2m 13.0d0 15.0d0 17.d0 2.d0 p))
	 (aa (m2l 5.0d0 6.0d0 7.d0 p mm))
	 (bb (m2l-v2 5.0d0 6.0d0 7.d0 p mm)))
    (format t "mm = ~a~%" mm)
    (format t "aa = ~a~%" aa)
    (format t "bb = ~a~%" bb)
    (reduce #'max (mapcar #'abs (mapcar #'- aa bb)))))
    
    
	   
