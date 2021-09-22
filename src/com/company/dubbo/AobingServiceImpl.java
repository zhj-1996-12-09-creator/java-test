package com.company.dubbo;

public class AobingServiceImpl implements AobingService{
    @Override
    public String hello(String name) {
        return "hello "+name;
    }
}
