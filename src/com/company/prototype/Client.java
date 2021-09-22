package com.company.prototype;

import java.util.Random;

/**原型模式
 * 把原来对象复制一份，新对象和原来一模一样，然后再去修改其中细节的地方
 */
public class Client {

    private static int MAX_COUNT = 6;

    public static void main(String[] args) {
        //模拟邮件发送
        int i=0;
        //模板定义出来
        AdvTemplate advTemplate = new AdvTemplate();
        Mail mail = new Mail(advTemplate);
        mail.setTail("xxx版权所有");
        while (i<MAX_COUNT){
            Mail cloneMail = mail.clone();
            cloneMail.setAppellation(getRandString(5)+" 先生（女士）");
            cloneMail.setReceiver(getRandString(5)+"@"+getRandString(8)+".com");
            //然后发送邮件
            sendMail(cloneMail);
            i++;
        }
    }
    public static void sendMail(Mail mail){
        System.out.println("标题："+mail.getSubject() + "\t收件人： "+mail.getReceiver()+"\t...发送成功！");
    }
    public static String getRandString(int maxLength){
        String source ="abcdefghijklmnopqrskuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        StringBuffer sb = new StringBuffer(); Random rand = new Random();
        for(int i = 0; i<maxLength; i++){
            sb.append(source.charAt(rand.nextInt(source.length())));
        }
        return sb.toString();
    }
}
