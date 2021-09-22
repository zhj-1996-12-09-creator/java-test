package com.company.parseXml;

import org.dom4j.*;
import org.dom4j.io.SAXReader;

import javax.xml.parsers.SAXParser;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.List;
import java.util.Map;

public class parseXml {
    static class SwordBaseCheckedException extends Throwable {
        private  String mess;
        public SwordBaseCheckedException(String _mess){
            this.mess = _mess;
        }
    }

    public static URL getConfigURL(String fileName, boolean noFoundThrowException) throws SwordBaseCheckedException {
        URL url = null;
        try {
            File file = new File(fileName);
            if (file.exists()) {
                url = file.toURI().toURL();
            } else {
                /*Map<String, Object> remoteConfig = SwordPlatformManager.getRemoteConfig();
                if (remoteConfig == null) {
                    url = null;
                } else {
                    url = new URL(remoteConfig.get("root-url") + fileName);
                }*/
            }
        } catch (MalformedURLException var14) {
            throw new SwordBaseCheckedException("构建配置文件路径时发生错误!");
        }

        if (url == null) {
            if (noFoundThrowException) {
                throw new SwordBaseCheckedException("配置文件" + fileName + "未找到 ");
            } else {
                return null;
            }
        } else {
            InputStream is = null;

            try {
                is = url.openStream();
            } catch (IOException var15) {
                if (!(var15 instanceof FileNotFoundException)) {
                    throw new SwordBaseCheckedException("读取配置文件" + url + "时发生错误! ");
                }

                if (noFoundThrowException) {
                    throw new SwordBaseCheckedException("配置文件" + url + "未找到 ");
                }

                url = null;
            } finally {
                if (is != null) {
                    try {
                        is.close();
                    } catch (IOException var13) {
                        throw new SwordBaseCheckedException("关闭配置文件" + url + "时发生错误! ");
                    }
                }

            }

            return url;
        }
    }

    public static void main(String[] args) throws MalformedURLException, DocumentException, SwordBaseCheckedException {
        SAXReader sreader = new SAXReader();
        String fileurl = "E:\\IDEAwork\\javaTest\\src\\com\\company\\parseXml\\parseXml.xml";
        URL url = getConfigURL(fileurl, true);
        //URL url = new URL(fileurl);
        Document document = sreader.read(url);
        Element element =document.getRootElement();
        QName QName_bean = element.getQName("bean");
        DocumentFactory DocumentFactory = QName_bean.getDocumentFactory();
        List list = DocumentFactory.getQNames();
        Map map = DocumentFactory.getXPathNamespaceURIs();
        // Element element_id = element_bean.element("id");
        //element_bean.getQName();
      //  System.out.println(element_bean.getStringValue());
        System.out.println(document.getXMLEncoding());
        System.out.println(document.getParent());
        System.out.println(document.getDocument());
        System.out.println(document.getNodeType());

    }
}
