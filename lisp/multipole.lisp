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

(defun e2e (p)
  (l2l -0.14135 -0.14135 -0.14135 p
       (l2l 0.93395 0.93395 0.93395 p
	    (m2l 9.3395 9.3395 9.3395 p
		 (m2m -1.8679 -1.8679 -1.8679 p
		      (m2m -0.93395 -0.93395 -0.93395 p
			   (p2m -0.43395 -0.43395 -0.43395 -1 p)))))))
