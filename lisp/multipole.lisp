(defun mult-idx (l m)
  (+ (/ (* l (1+ l)) 2) m))

(defun mult-get (mult l m)
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
			    (I^m_l (1- m) (1- m) x y z))))
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
  (let ((ips (p2m dx dy dz 1 p)))
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
  (let ((tmp (p2l dx dy dz 1 p)))
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
  (let ((tmp (p2m dx dy dz 1 p)))
    (mapcan
     #'identity
     (loop for n from 0 to p collect
	  (loop for m from 0 to n collect
	       (loop for k from 0 to (- p n) sum
		    (loop for l from (- k) to k sum
			 (* (conjugate (mult-get tmp k l))
			    (mult-get loc (+ n k) (+ m l))))))))))

  
	    
    
