package com.company.Collections;

import java.util.concurrent.*;

/**
 * 减法计数器
 */
public class CountDownLatchDemo {

   /* public static void main(String[] args) throws InterruptedException {
        CountDownLatch countDownLatch = new CountDownLatch(6);
        for (int i = 1; i <= 5 ; i++) {
           new Thread(()->{
               System.out.println(Thread.currentThread().getName()+"===执行了");
               countDownLatch.countDown();//自减
           }).start();
        }
        //阻塞等待计数器归零
        countDownLatch.await();
        System.out.println("执行结束");
    }*/

    /**
     * 加法计数器
     * @param args
     */
    /*public static void main(String[] args) {
        CyclicBarrier cyclicBarrier = new CyclicBarrier(7,()->{ System.out.println("召唤神龙成功"); });

        for (int i = 0; i < 7; i++) {
            final int tempInt = i;
            new Thread(()->{
                try {
                    System.out.println(Thread.currentThread().getName()+"收集了 第"+ tempInt +"颗龙珠");
                    cyclicBarrier.await();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (BrokenBarrierException e) {
                    e.printStackTrace();
                }
            }).start();
        }

    }*/

    //Semaphore,信号灯
    //信号量主要用于两个目的：一个是用于多个共享资源的互斥使用，另一个用于并发线程数的控制。
    public static void main(String[] args) {
        Semaphore semaphore = new Semaphore(3);
        for (int i = 0; i < 6; i++) {
            new Thread(()->{
                try {
                    semaphore.acquire();
                    System.out.println(Thread.currentThread().getName()+" 抢 到了车位");
                    TimeUnit.SECONDS.sleep(2);
                    System.out.println(Thread.currentThread().getName()+"离开车位");
                }catch (InterruptedException e){
                    e.printStackTrace();
                }
                finally {
                    semaphore.release();
                }
            },""+i).start();
        }
    }



}
