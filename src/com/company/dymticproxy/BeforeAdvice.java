package com.company.dymticproxy;

public class BeforeAdvice implements Advice{
    @Override
    public void exec() {
        System.out.println("前置通知");
    }

    @Override
    public void afterAdvice() {
        System.out.println("后置通知");
    }

}
