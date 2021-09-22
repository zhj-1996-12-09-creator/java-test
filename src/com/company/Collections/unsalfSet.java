package com.company.Collections;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArraySet;

/**
 * CopyOnWriteArrayList为什么并发安全且性能比Vector好
 * 我知道Vector是增删改查方法都加了synchronized，保证同步，但是每个方法执行的时候都要去获得
 * 锁，性能就会大大下降，而CopyOnWriteArrayList 只是在增删改上加锁，但是读不加锁，在读方面的性
 * 能就好于Vector，CopyOnWriteArrayList支持读多写少的并发情况。
 */
public class unsalfSet {

    //public static Set<String> set = new HashSet<>();
    //Exception in thread "Thread-8" java.util.ConcurrentModificationException
    //public static Set<String> set = Collections.synchronizedSet(new HashSet<>());
    public static Set<String> set = new CopyOnWriteArraySet<>();
    public static void main(String[] args) {
        for (int i = 0; i < 30; i++) {
            int temp = i;
            new Thread(()->{
                set.add(""+temp);
                System.out.println(set);
            }).start();
        }

    }
}
