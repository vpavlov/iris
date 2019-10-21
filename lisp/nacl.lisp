(defconstant +NaCl-lattice-constant+ 5.6402d0) ;; [ang]
(defconstant +NaCl-Madelung-constant+ 1.7475646d0)

(defun nacl (n)
  (when (= (mod n 2) 1)
    (error "N must be even!"))
  (let* ((n/2 (/ n 2))
	 (a +NaCl-lattice-constant+)
	 (a/2 (/ a 2))
	 (L (* n/2 a))
	 (retval (make-array (list (* n n n) 4)))
	 (nn 0)
	 (theoretical-energy (/ (* -1.0d0 +NaCl-Madelung-constant+ n n n)
				+NaCl-lattice-constant+)))
    (dotimes (i n/2)
      (let ((x (* i a)))
	(dotimes (j n/2)
	  (let ((y (* j a)))
	    (dotimes (k n/2)
	      (let ((z (* k a)))
		(setf
		 (aref retval nn 0) x
		 (aref retval nn 1) y
		 (aref retval nn 2) z
		 (aref retval nn 3) 1.0d0
		 
		 (aref retval (+ nn 1) 0) (+ x a/2)
		 (aref retval (+ nn 1) 1) y
		 (aref retval (+ nn 1) 2) z
		 (aref retval (+ nn 1) 3) -1.0d0
		 
		 (aref retval (+ nn 2) 0) x
		 (aref retval (+ nn 2) 1) (+ y a/2)
		 (aref retval (+ nn 2) 2) z
		 (aref retval (+ nn 2) 3) -1.0d0
		 
		 (aref retval (+ nn 3) 0) (+ x a/2)
		 (aref retval (+ nn 3) 1) (+ y a/2)
		 (aref retval (+ nn 3) 2) z
		 (aref retval (+ nn 3) 3) 1.0d0

		 (aref retval (+ nn 4) 0) x
		 (aref retval (+ nn 4) 1) y
		 (aref retval (+ nn 4) 2) (+ z a/2)
		 (aref retval (+ nn 4) 3) -1.0d0
		 
		 (aref retval (+ nn 5) 0) (+ x a/2)
		 (aref retval (+ nn 5) 1) y
		 (aref retval (+ nn 5) 2) (+ z a/2)
		 (aref retval (+ nn 5) 3) 1.0d0
		 
		 (aref retval (+ nn 6) 0) x
		 (aref retval (+ nn 6) 1) (+ y a/2)
		 (aref retval (+ nn 6) 2) (+ z a/2)
		 (aref retval (+ nn 6) 3) 1.0d0
		 
		 (aref retval (+ nn 7) 0) (+ x a/2)
		 (aref retval (+ nn 7) 1) (+ y a/2)
		 (aref retval (+ nn 7) 2) (+ z a/2)
		 (aref retval (+ nn 7) 3) -1.0d0
		 nn (+ nn 8)
		 )))))))
    (values retval L theoretical-energy)))

(defun nacl-pdb (n fname)
  (multiple-value-bind (charges l e0)
      (nacl n)
    (declare (ignore e0))
    (with-open-file (f fname :direction :output :if-exists :supersede)
      (format f "CRYST1~9,3f~9,3f~9,3f  90.00  90.00  90.00 P 1           1~%" l l l)
      (dotimes (i (array-dimension charges 0))
	(format f "ATOM  ~5d ~a     X   1    ~8,3f~8,3f~8,3f  0.00  0.00            ~%"
		(1+ i)
		(if (= (aref charges i 3) 1.0d0) " Na+" " Cl-")
		(aref charges i 0)
		(aref charges i 1)
		(aref charges i 2)))
      (format f "END~%"))))
