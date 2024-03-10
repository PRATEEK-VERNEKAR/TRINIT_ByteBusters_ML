const mongoose=require('mongoose');
const express=require('express');
const dotenv=require('dotenv');
const cookieParser = require('cookie-parser');
const axios = require('axios');
const cors = require('cors');

require('./db/connection');

const Text=require('./models/schema');
const tweet1=require('./tweets1_data')

const app=express();

app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());

// app.use(Router);

// // require('dotenv').config();

// dotenv.config({path:'./config.env'});                                       //Written only once in app.js

// const PORT=process.env.PORT;

app.use(cors());

// const middleware=(req,res,next)=>{                  //passed as parameter in the app.get() and first it runs then next() runs the
//     console.log("Middleware Part");                 //app.get() route
//     next();
// }


app.get('/',(req,res)=>{
    console.log("Home page");
    res.send(tweet1);
})

app.get('/predict',async (req,res)=>{
    try{

        tweet1.forEach(async (tweet)=>{
            const modelRes=await axios.post("http://127.0.0.1:5000/predict",{
                text:tweet.text
            });
            console.log(modelRes.data);

            const existingText = await Text.findOne({ name: tweet.name });


            const datasplit=tweet.date.split(' · ');

            console.log(datasplit);
            if(existingText){
                existingText.dataArray.push({
                    link: tweet.link,
                    text: tweet.text,
                    date: datasplit[0],
                    time: datasplit[1],
                    commenting: modelRes.model2,
                    staring: modelRes.model3,
                    groping: modelRes.model4,
                    severity:modelRes.severity
                });

                await existingText.save();
            }
            else{
                const newText = new Text({
                    name: tweet.name,
                    dataArray: [
                        {
                            link: tweet.link,
                            text: tweet.text,
                            date: datasplit[0],
                            time: datasplit[1],
                            commenting: modelRes.model2,
                            staring: modelRes.model3,
                            groping: modelRes.model4,
                            severity:modelRes.severity
                        },
                    ],
                });

                await newText.save();
            }
        })

        return res.status(200).json({message:"Predicted and updated successfully"});

    }
    catch(err){
        return res.status(500).json({message:"Internal Server Error"});
    }
})

app.get('/top_severe',async(req,res)=>{
    try{
        const topSevereDocuments = await Text.find({}).sort({ 'dataArray.severity': -1 }).limit(3);
        return res.status(200).json(topSevereDocuments);
    }
    catch(err){
        console.log(err);
        return res.status(500).json({message:"Internal Server Error"});
    }
})


app.get('/top_severe',async(req,res)=>{
    try{
        const topSevereDocuments = await Text.find({}).sort({ 'dataArray.severity': -1 }).limit(3);
        return res.status(200).json(topSevereDocuments);
    }
    catch(err){
        console.log(err);
        return res.status(500).json({message:"Internal Server Error"});
    }
})

app.post('/name_extract',async(req,res)=>{
    try{
        const name=req.body;
        console.log(name);
        const nameTweets=await Text.find({name:name.name})
        return res.status(200).json(nameTweets);
    }
    catch(err){
        console.log(err);
        return res.status(500).json({message:"Internal Server Error"})
    }
})


app.post('/date_extract',async (req,res)=>{
    try{
        const {date}=req.body;
        console.log(date);

        const dateTweets=await Text.find({
            'dataArray.date':date,
        })

        return res.status(200).json(dateTweets);
    }
    catch(err){
        return res.status(200).json({message:"Internal Server Error"})
    }
})


app.listen(8000,()=>{
    console.log(`Running on port number ${8000}`);
})