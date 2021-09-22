package com.company.lamda;

import java.util.ArrayList;
import java.util.List;

public class lambdaTest {

    private void testAnonymousInnerClass(lambda lambda) {
        Integer number = 1;
        boolean result = lambda.test01(number);
        System.out.println(result);
    }

    public static void main(String[] args) {
       /* lambdaTest lambdaTest = new lambdaTest();
        lambdaTest.testAnonymousInnerClass(number -> {
            if (number >1){
                System.out.println("大于1");
                return true;
            }else{
                System.out.println("小于等于1");
            }
            return false;
        });*/
        // Java7
       /* new Thread(new Runnable() {
            @Override
            public void run() {
                for (int i = 0; i < 100; i++) {
                    System.out.println("java7线程"+i);
                }
            }
        }).start();*/

        // Java8
       /* new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                System.out.println("java8线程"+i);
            }
        }).start();*/
        List<String> list = new ArrayList<>();
        list.add("ceshi1");
        list.add("ceshi2");
        list.add("ceshi3");
        list.forEach(i -> {
            i=i.substring(0,5);
            System.out.println(i);
        });


    }
}
