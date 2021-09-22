package com.company.Collections;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

public class BlockingQueueDemo {

   /* //有异常  add   remove   elment
    public static void main(String[] args) {
        BlockingQueue<Integer> queue = new ArrayBlockingQueue<Integer>(3);
        queue.add(1);
        queue.add(2);
        queue.add(3);
        System.out.println(queue.element());
        //queue.add(4);//Exception in thread "main" java.lang.IllegalStateException: Queue full
        queue.remove();
        queue.remove();
        queue.remove();
        queue.remove();//Exception in thread "main" java.util.NoSuchElementException
    }*/

    /*//无异常，有返回
    public static void main(String[] args) {
        BlockingQueue<Integer> queue = new ArrayBlockingQueue<Integer>(3);
        queue.offer(1);
        queue.offer(2);
        queue.offer(3);
        System.out.println(queue.offer(4));//false

        queue.poll();
        queue.poll();
        queue.poll();
        System.out.println(queue.poll());//null
    }*/

   /* //一致阻塞
    public static void main(String[] args) throws InterruptedException {
        BlockingQueue<Integer> queue = new ArrayBlockingQueue<Integer>(3);
        queue.put(1);
        queue.put(2);
        queue.put(3);
        //queue.put(4);
        System.out.println("-------------------");
        queue.take();queue.take();
        queue.take();
       // queue.take();
        System.out.println("ok" +
                "");

    }*/

    //超时等待
    public static void main(String[] args) throws InterruptedException {
        BlockingQueue<Integer> queue = new ArrayBlockingQueue<Integer>(3);
        queue.offer(1);
        queue.offer(2);
        queue.offer(3);
        queue.offer(4,2, TimeUnit.SECONDS);
        System.out.println("sdjsa ");
        queue.poll();
        queue.poll();
        queue.poll();
        queue.poll(2,TimeUnit.SECONDS);

    }
}
