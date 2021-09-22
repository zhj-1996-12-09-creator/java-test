package com.company.ChainOfResponsibility;

public class Women implements IWomen{
    private int type;
    private String request = "";
    public Women(int _type,String _request){
        this.type = _type;
        this.request = _request;
    }
    @Override
    public int getType() {
        return type;
    }

    @Override
    public String getRequest() {
        return request;
    }
}
