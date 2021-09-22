package com.company.Collections;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class unsalfList {

    //public static List<String> list = new ArrayList<>();
   // public static List<String> list = new CopyOnWriteArrayList<>();
    public static List<String> list = Collections.synchronizedList(new ArrayList<>());
    public static void main(String[] args) {
        for (int i = 1; i <= 30; i++) {
            int temp = i;
            new Thread(()->{
                list.add(""+temp);
                System.out.println(list);
            },temp+"").start();
        }
        System.out.println(list.size());
    }
}
