(defvar *eus-cuda-matrix-library*
  (load-foreign "../build/libeus_cuda_matrix.so"))

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

(defun test-matrixMul
  ()
  (let* ((row-a 32)
         (col-a 32)
         (col-b 64)
         (mat-a
          (make-array (list row-a col-a) :element-type :float :initial-element 1))
         (mat-b
          (make-array (list col-a col-b) :element-type :float :initial-element 1))
         (mat-c
          (make-array (list row-a col-b) :element-type :float))
         )

    (_call-matrixMultiply row-a col-a col-b (mat-a . entity) (mat-b . entity) (mat-c . entity))

    (print "A=")
    (format-array mat-a)
    (print "B=")
    (format-array mat-b)
    (print "C=")
    (format-array mat-c)

    nil
    ))
(warn "(test-matrixMul)~%")

(defun test-matrixMul-large
  ()
  (let* ((row-a 320)
         (col-a 1280)
         (col-b 2560)
         (mat-a
          (make-array (list row-a col-a) :element-type :float :initial-element 1.5))
         (mat-b
          (make-array (list col-a col-b) :element-type :float :initial-element -pi/2))
         (cuda-result
          (make-array (list row-a col-b) :element-type :float))
         (eus-result
          (make-array (list row-a col-b) :element-type :float))
         (max-error)
         )
    (warning-message 2 "cuda:~%")
    (bench
     (_call-matrixMultiply row-a col-a col-b (mat-a . entity) (mat-b . entity) (cuda-result . entity)))

    (warning-message 2 "eus:~%")
    (bench
     (m* mat-a mat-b eus-result))

    (setq max-error (apply #'max (map cons #'abs (array-entity (m- cuda-result eus-result)))))
    (warning-message 2 "error = ~a~%" max-error)

    nil
    ))
(warn "(test-matrixMul-large)~%")
