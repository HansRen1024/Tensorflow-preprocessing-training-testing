# Tensorflow-preprocessing-training-testing

(non-distribution & distribution)


Use Tensorflow to do classification containing data preparation, training, testing.(single computer single GPU &amp; single computer multi-GPU &amp; multi-computer multi-GPU)

---

中文博客地址： http://blog.csdn.net/renhanchi/article/details/79570665

---

**All parameters are in `arg_parsing.py`. So before you start this program, you should read it carefully!**

**STEPS:**

1. Put all images in different diractories. Then run **img2list.sh** to create a txt file containing pathes and labels of all iamges.

    ![txt content](https://img-blog.csdn.net/20180320151535236 "")

2. Run **list2bin.py** to convert the images from rgb to tfrecords.

3. For single computer, one GPU or more, whatever. Just run:

        python main.py
  
4. For distribution, first you should modify **PS_HOSTS** and **WORKER_HOSTS** in **arg_parsing.py**. And then copy all dataset and codes to every server. 

  For ps host, run:

    CUDA_VISIBLE_DEVICES='' python src/main.py --job_name=ps --task_index=0

  **CUDA_VISIBLE_DEVICES=''** means using CPU to concat parameters.

  For worker host, run:

    python src/main.py --job_name=worker --task_index=0

  Do remenber to increase **task_index** in every server.

5. All ckpt and event files will be saved in **MODEL_DIR**.
6. For testing, just run:

       python src/main.py --mode=testing

---

**Notes**

1. DO READ **arg_parsing.py** again and again to understand and control this program.

2. Use **CUDA_VISIBLE_DEVICES=0,2** to choose GPUs.

3. For visualization, run:

       tensorboard --logdir=models/
    
4. More details, please see my blog above.
