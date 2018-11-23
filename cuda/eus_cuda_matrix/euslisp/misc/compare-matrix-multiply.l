(load "../eus-cuda-matrix.l")


(defun cuda-m* (a b c)
  (let ((row-a (array-dimension a 0))
        (col-a (array-dimension a 1))
        (col-b (array-dimension b 1))
        )
    (_call-matrixMultiply
     row-a col-a col-b
     (a . entity)
     (b . entity)
     (c . entity)
     )
    c)
  )

(defun cuda-cublas-m* (a b c)
  (let ((row-a (array-dimension a 0))
        (col-a (array-dimension a 1))
        (col-b (array-dimension b 1))
        )
    (_call-matrixMultiplyCUBLAS
     row-a col-a col-b
     (a . entity)
     (b . entity)
     (c . entity)
     )
    c)
  )

(defun cblas-m* (a b c &optional (alpha 1.0) (beta 1.0))
  (let ((major 101) ;; 101: rowmajor, 102: colmajor
        (trans 111) ;; 111: notrans, 112: trans, 113: conjtrans
        (m (array-dimension a 0))
        (n (array-dimension b 1))
        (k (array-dimension a 1))
        )
    (cblas_dgemm
     major trans trans
     m n k alpha
     (a . entity) k
     (b . entity) n
     beta
     (c . entity) n)
    c)
  )

(defun openblas-m* (a b c &optional (alpha 1.0) (beta 1.0))
  (let ((major 101) ;; 101: rowmajor, 102: colmajor
        (trans 111) ;; 111: notrans, 112: trans, 113: conjtrans
        (m (array-dimension a 0))
        (n (array-dimension b 1))
        (k (array-dimension a 1))
        )
    (openblas_dgemm
     major trans trans
     m n k alpha
     (a . entity) k
     (b . entity) n
     beta
     (c . entity) n)
    c)
  )


(defun compare-matrix-multiply
  ()
  (let* ((row-a 320)
         (col-a 1280)
         (col-b 2560)
         (a
          (make-array (list row-a col-a) :element-type :float :initial-element 1.5))
         (b
          (make-array (list col-a col-b) :element-type :float :initial-element -pi/2))
         (cuda-result
          (make-array (list row-a col-b) :element-type :float))
         (cuda-cublas-result
          (make-array (list row-a col-b) :element-type :float))
         (cblas-result
          (make-array (list row-a col-b) :element-type :float))
         (openblas-result
          (make-array (list row-a col-b) :element-type :float))
         (eus-result
          (make-array (list row-a col-b) :element-type :float))
         (max-error)
         )
    (warning-message 2 "cuda:~%")
    (bench (cuda-m* a b cuda-result))

    (warning-message 2 "cuda-cublas:~%")
    (bench (cuda-cublas-m* a b cuda-cublas-result))

    (warning-message 2 "cblas:~%")
    (bench (cblas-m* a b cblas-result))

    (warning-message 2 "openblas:~%")
    (bench (openblas-m* a b openblas-result))

    (warning-message 2 "eus:~%")
    (bench (m* a b eus-result))

    (setq max-error (apply #'max (map cons #'abs (array-entity (m- cuda-result eus-result)))))
    (warning-message 2 "error between cuda and eus: ~a~%" max-error)
    (setq max-error (apply #'max (map cons #'abs (array-entity (m- cuda-cublas-result eus-result)))))
    (warning-message 2 "error between cuda-cublas and eus: ~a~%" max-error)
    (setq max-error (apply #'max (map cons #'abs (array-entity (m- cblas-result eus-result)))))
    (warning-message 2 "error between cblas and eus: ~a~%" max-error)
    (setq max-error (apply #'max (map cons #'abs (array-entity (m- openblas-result eus-result)))))
    (warning-message 2 "error between openblas and eus: ~a~%" max-error)

    nil
    ))
(warn "(compare-matrix-multiply)~%")
