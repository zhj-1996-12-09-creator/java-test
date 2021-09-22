package com.company;

import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;

public class test {

    /**
     * 可重入锁的验证
     * @param args
     */
    public static void mi(){
        new Thread(()->{
            synchronized(test.class){
                System.out.println("外层锁");
                synchronized (test.class){
                    System.out.println("中层锁");
                    synchronized (test.class){
                        System.out.println("内层锁");
                    }
                }
            }
        },"l").start();

    }

    public static void ij(){
        Thread a = new Thread(()->{
            System.out.println("A进入");
            LockSupport.park();
            System.out.println("A被唤醒"
            );
        },"A");
        a.start();
        new Thread(()->{
            LockSupport.unpark(a);
            System.out.println("b去唤醒A");
        }).start();
    }




    public static void main(String[] args) {
        String s1 = new StringBuilder("ap").append("asss").toString();
        System.out.println(s1 == s1.intern());
        String s2 = "apasss";
        System.out.println(s1.intern() == s2.intern());
        //看下字符串java
        String s3 = "java";
        String s4 = new StringBuilder("java").toString();
        System.out.println(s3 == s3.intern());
        System.out.println(s4 == s4.intern());//为甚麽式false呢？因为在sun.misc.version类中，已经加载过这个常量“java”，
        //它是位于rt。jar，那时bootStrapclassloader加载的，肯定优先
        ij();
    }

    /**
     * leetcode
     *多线程第一题，lockSupport+自旋锁
     */
    class Foo {

        public Foo() {

        }
        Thread a = null;
        Thread b = null;
        Thread c = null;

        public void first(Runnable printFirst) throws InterruptedException {
            a = Thread.currentThread();
            // printFirst.run() outputs "first". Do not change or remove this line.
            printFirst.run();
            while (b == null){

            }
            LockSupport.unpark(b);

        }

        public void second(Runnable printSecond) throws InterruptedException {
            b = Thread.currentThread();
            // printSecond.run() outputs "second". Do not change or remove this line.
            LockSupport.park();
            printSecond.run();
            while (c == null){

            }
            LockSupport.unpark(c);
        }

        public void third(Runnable printThird) throws InterruptedException {
            c = Thread.currentThread();
            LockSupport.park();
            // printThird.run() outputs "third". Do not change or remove this line.
            printThird.run();
        }
    }


    ReentrantLock lock = new ReentrantLock();


    public void aqs(){
        new Thread(()->{
            lock.lock();
            try {
                System.out.println("A 进入到执行");
                try {
                    TimeUnit.MILLISECONDS.sleep(20L);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } finally {
                lock.unlock();
            }
        },"A").start();


        new Thread(()->{
            lock.lock();
            try {
                System.out.println("B 进入到执行");

            } finally {
                lock.unlock();
            }
        },"B").start();

        new Thread(()->{
            lock.lock();
            try {
                System.out.println("C 进入到执行");

            } finally {
                lock.unlock();
            }
        },"C").start();

    }

}
