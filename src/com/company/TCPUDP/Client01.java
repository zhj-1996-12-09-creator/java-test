package com.company.TCPUDP;

import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Scanner;

public class Client01 {
    public static void main(String[] args) throws IOException {

        //创建套接字和并封装了ip与端口                                                                                            7
        try {
            Socket socket = new Socket("127.0.0.1", 8080);
            //根据socket获得输出流
            OutputStream out = socket.getOutputStream();
            //控制台输入以IO的形式发送到服务器
            System.out.println("TCP连接成功 \n请输入：");
            while (true) {
                byte[] car = new Scanner(System.in).nextLine().getBytes();
                out.write(car);
                System.out.println("TCP协议的Socket发送成功");
                out.flush();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {

        }
    }}