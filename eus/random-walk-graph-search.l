;; (load "package://jskeus/irteus/irtgraph.l")

(defclass random-walk-node
  :super node
  :slots (value)
  )

(defmethod random-walk-node
  (:init
   (name
    tmp-value)
   (send-super :init name)
   (setq value tmp-value)
   )
  (:value
   ()
   value
   )
  (:successors
   (&rest args)
   (let* ((new-node-list)
          (new-node-num 10)
          (dummy-arc (instance costed-arc :init))
          )
     (dotimes (i new-node-num)
       (let* ((rand-value (- (random 10) 5))
              (new-value (+ (send self :value) rand-value))
              (new-name (read-from-string (format nil ":value=~a" new-value)))
              (new-node (instance random-walk-node :init new-name new-value))
              )
         (push new-node new-node-list)
         ))
     (mapcar #'(lambda (n) (cons dummy-arc n)) new-node-list)
     ))
  )

(defmethod costed-arc
  (:init
   (&rest args)
   (setq cost 1.0)
   )
  )

(defclass random-walk-graph
  :super graph
  :slots ()
  )

(defmethod random-walk-graph
  (:goal-test
   (n)
   (> (send n :value) 10)
   )
  )

(defun test-random-walk-graph-search
  (&key
   (solver-type
    ;; :depth-first
    ;; :breadth-first
    ;; :best-first
    :a*
    )
   )
  (defmethod a*-graph-search-solver
    (:hn
     (n p)
     (- (send (send n :state) :value))
     ))
  (let* ((gr (instance random-walk-graph :init))
         (solver
          (case solver-type
                (:depth-first
                 (instance depth-first-graph-search-solver :init))
                (:breadth-first
                 (instance breadth-first-graph-search-solver :init))
                (:best-first
                 (instance best-first-graph-search-solver :init gr))
                (:a*
                 (instance a*-graph-search-solver :init gr))
                ))
         path
         )
    (send gr :add-node (instance random-walk-node :init :start 0))
    (send gr :start-state (send gr :node :start))
    (setq path (send solver :solve gr :verbose nil))
    (pprint (send-all (send-all path :state) :name))
    path
    ))

(warn "(test-random-walk-graph-search)~%")
