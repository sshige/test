(defvar *eus-cuda-matrix-library* (load-foreign "../../build/libeus_cuda_matrix.so"))

(defvar *cblas-library* (load-foreign (format nil "/usr/lib/libcblas.so")))

;; int call_matrixMultiply(int row_A, int col_A, int col_B, double *h_A, double *h_B, double *h_C);
(defforeign _call-matrixMultiply
  *eus-cuda-matrix-library*
  "call_matrixMultiply"
  (
   :integer
   :integer
   :integer
   :string
   :string
   :string
   )
  :integer
  )

;; int call_matrixMultiplyCUBLAS(int row_A, int col_A, int col_B, double *h_A, double *h_B, double *h_C);
(defforeign _call-matrixMultiplyCUBLAS
  *eus-cuda-matrix-library*
  "call_matrixMultiplyCUBLAS"
  (
   :integer
   :integer
   :integer
   :string
   :string
   :string
   )
  :integer
  )

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

;;  ( CBLAS_LAYOUT CBLAS_TRANSPOSE CBLAS_TRANSPOSE :integer :integer :integer :float :integer :integer :integer :integer :float :integer :integer)
(defforeign cblas_dgemm
  *cblas-library*
  "cblas_dgemm"
  ()
  :integer
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

    (warning-message 2 "eus:~%")
    (bench (m* a b eus-result))

    (setq max-error (apply #'max (map cons #'abs (array-entity (m- cuda-result eus-result)))))
    (warning-message 2 "error between cuda and eus: ~a~%" max-error)
    (setq max-error (apply #'max (map cons #'abs (array-entity (m- cuda-cublas-result eus-result)))))
    (warning-message 2 "error between cuda-cublas and eus: ~a~%" max-error)
    (setq max-error (apply #'max (map cons #'abs (array-entity (m- cblas-result eus-result)))))
    (warning-message 2 "error between cblas and eus: ~a~%" max-error)

    nil
    ))
(warn "(compare-matrix-multiply)~%")
