


(defun get-measured-force
  (tm)
  (cond ((< tm 700)
         (float-vector (sin (/ tm 100.0))
                       (sin (+ (/ tm 100.0) (* pi 0.5)))
                       (sin (+ (/ tm 100.0) (* pi 1.0))))
         )
        (t
         (float-vector 0 0 0)
         ))
  )

(defun update-target-force-without-diff-for-door
  (current-target-force
   current-measured-force)
  (let* ((ex (float-vector 1 0 0))
         (ey (float-vector 0 1 0))
         (ez (float-vector 0 0 1))
         )
    (v+
     (v+ (scale (v. current-measured-force ex) ex)
         (scale (v. current-target-force ey) ey))
     (scale (v. current-target-force ez) ez))
    ))

(defun update-target-force-for-door
  (current-target-force
   current-measured-force
   &key
   (gain 0.02))
  (let* ((ex (float-vector 1 0 0))
         )
    (v+ (scale gain (scale (v. (v- current-measured-force current-target-force) ex) ex))
        current-target-force)
    ))


(defun test
  (&optional (update-func #'update-target-force-without-diff-for-door))
  (let* ((target-force (float-vector 0 0 0))
         measured-force
         )
    (dotimes (tm 1000)
      (setq measured-force (get-measured-force tm))
      (setq target-force
            (funcall update-func
                     target-force
                     measured-force))
      (dolist (x (concatenate cons target-force measured-force))
        (format t "~a " x))
      (format t "~%")
      )
    ))

#|
(test #'update-target-force-without-diff-for-door)
(test #'update-target-force-for-door)
