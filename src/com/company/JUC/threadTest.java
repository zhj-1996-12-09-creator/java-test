package com.company.JUC;

import java.util.concurrent.*;
import java.util.concurrent.locks.LockSupport;

public class threadTest {

    private BlockingQueue<Integer> blockingQueue = new SynchronousQueue<Integer>();

    public  void provide() throws InterruptedException {
        int i =0;
        while(true) {
            blockingQueue.put(i++);
        }


    }

    public  void consunmer() throws InterruptedException {
        while(true){
            System.out.println(blockingQueue.take());
        }
    }


    public static void main(String[] args) {
        /*Semaphore semaphore = new Semaphore(3);

        for (int i = 0; i < 6 ; i++) {
            new Thread(()->{
                try {
                    semaphore.acquire();
                    System.out.println(Thread.currentThread().getName() +"已经获得车位");
                    TimeUnit.SECONDS.sleep(3);
                    System.out.println(Thread.currentThread().getName() +"已经离开车位");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }finally {
                    semaphore.release();
                }
            },String.valueOf(i)).start();
        }*/
        threadTest threadTest = new threadTest();
        //生产者
        new Thread(()->{
            try {
                threadTest.provide();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
        new Thread(()->{
            try {
                threadTest.consunmer();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
    }
}
