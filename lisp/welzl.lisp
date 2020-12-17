(defun dot (a b)
  (reduce #'+ (mapcar #'* a b)))

(defun cross (a b)
  (list
   (- (* (nth 1 a) (nth 2 b)) (* (nth 2 a) (nth 1 b)))
   (- (* (nth 2 a) (nth 0 b)) (* (nth 0 a) (nth 2 b)))
   (- (* (nth 0 a) (nth 1 b)) (* (nth 1 a) (nth 0 b)))))
   

(defun sphere (a b c d)
  (let* ((b1 (mapcar #'- b a))
	 (c1 (mapcar #'- c a))
	 (d1 (mapcar #'- d a))
	 (b1^2 (dot b1 b1))
	 (c1^2 (dot c1 c1))
	 (d1^2 (dot d1 d1))
	 (2b1 (mapcar #'(lambda (x) (* 2 x)) b1))
	 (denom (dot 2b1 (cross c1 d1)))
	 (v1 (mapcar #'(lambda (x) (* b1^2 x)) (cross c1 d1)))
	 (v2 (mapcar #'(lambda (x) (* c1^2 x)) (cross d1 b1)))
	 (v3 (mapcar #'(lambda (x) (* d1^2 x)) (cross b1 c1)))
	 (v123 (mapcar #'+ v1 v2 v3))
	 (i1 (mapcar #'(lambda (x) (/ x denom)) v123)))
    (list (mapcar #'+ i1 a) (sqrt (dot i1 i1)))))
    
    
    
